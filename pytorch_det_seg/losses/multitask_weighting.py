import torch

try:
    from LibMTL.weighting import CAGrad, DWA, EW, GLS, GradDrop, GradNorm, GradVac, IMTL, MGDA, PCGrad, RLW, UW
    LIBMTL_AVAILABLE = True
except Exception:
    CAGrad = DWA = EW = GLS = GradDrop = GradNorm = GradVac = IMTL = MGDA = PCGrad = RLW = UW = None
    LIBMTL_AVAILABLE = False


def _safe_reset_param_grads(owner, new_grads):
    count = 0
    for param in owner.get_share_params():
        beg = 0 if count == 0 else sum(owner.grad_index[:count])
        end = sum(owner.grad_index[:(count + 1)])
        grad_view = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
        if param.grad is None:
            param.grad = grad_view
        else:
            param.grad.data = grad_view
        count += 1


class LegacyEWLikeWeighting:
    def __init__(self, config, model):
        self.device = torch.device(config.device)
        self.task_num = 2
        self.task_name = ['det_loss', 'seg_loss']
        self.get_share_params = lambda: model.backbone.parameters()
        self.zero_grad_share_params = model.backbone.zero_grad

    def init_param(self):
        return None

    def backward(self, losses, **kwargs):
        loss = torch.mul(losses, torch.ones_like(losses).to(self.device)).sum()
        loss.backward()
        return [1.0, 1.0]


class SafeGradDrop(GradDrop):
    def backward(self, losses, **kwargs):
        leak = kwargs['leak']
        if not self.rep_grad:
            raise ValueError('No support method GradDrop with parameter gradients (rep_grad=False)')
        grads = torch.autograd.grad(losses.sum(), self.rep, retain_graph=False, create_graph=False)[0]
        transformed_grad = grads * (leak + (1 - leak) * (grads.sign() != 0).to(grads.dtype))
        self.rep.backward(transformed_grad)
        return None


    def _reset_grad(self, new_grads):
        count = 0
        for param in self.get_share_params():
            beg = 0 if count == 0 else sum(self.grad_index[:count])
            end = sum(self.grad_index[:(count + 1)])
            grad_view = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
            if param.grad is None:
                param.grad = grad_view
            else:
                param.grad.data = grad_view
            count += 1


class SafePCGrad(PCGrad):
    def _reset_grad(self, new_grads):
        _safe_reset_param_grads(self, new_grads)


class SafeMGDA(MGDA):
    def _reset_grad(self, new_grads):
        _safe_reset_param_grads(self, new_grads)


class SafeCAGrad(CAGrad):
    def _reset_grad(self, new_grads):
        _safe_reset_param_grads(self, new_grads)

    def backward(self, losses, **kwargs):
        calpha, rescale = kwargs['calpha'], kwargs['rescale']
        if self.rep_grad:
            raise ValueError('No support method CAGrad with representation gradients (rep_grad=True)')
        self._compute_grad_dim()
        grads = self._compute_grad(losses, mode='backward')

        GG = torch.matmul(grads, grads.t()).cpu()
        g0_norm = (GG.mean() + 1e-8).sqrt()

        x_start = __import__('numpy').ones(self.task_num) / self.task_num
        bnds = tuple((0, 1) for _ in x_start)
        cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
        A = GG.numpy()
        b = x_start.copy()
        c = (calpha * g0_norm + 1e-8).item()

        def objfn(x):
            x_row = x.reshape(1, self.task_num)
            b_col = b.reshape(self.task_num, 1)
            x_col = x.reshape(self.task_num, 1)
            return (x_row.dot(A).dot(b_col) + c * (__import__('numpy').sqrt(x_row.dot(A).dot(x_col) + 1e-8))).sum()

        from scipy.optimize import minimize
        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.tensor(w_cpu, device=self.device, dtype=grads.dtype)
        gw = (grads * ww.view(-1, 1)).sum(0)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-8)
        g = grads.mean(0) + lmbda * gw
        if rescale == 0:
            new_grads = g
        elif rescale == 1:
            new_grads = g / (1 + calpha ** 2)
        elif rescale == 2:
            new_grads = g / (1 + calpha)
        else:
            raise ValueError('No support rescale type {}'.format(rescale))
        self._reset_grad(new_grads)
        return w_cpu


class SafeGradVac(GradVac):
    def _reset_grad(self, new_grads):
        _safe_reset_param_grads(self, new_grads)


class SafeIMTL(IMTL):
    def _reset_grad(self, new_grads):
        _safe_reset_param_grads(self, new_grads)


_WEIGHTING_REGISTRY = {
    'EW': EW if LIBMTL_AVAILABLE else LegacyEWLikeWeighting,
    'GradNorm': GradNorm,
    'MGDA': SafeMGDA if MGDA is not None else None,
    'UW': UW,
    'DWA': DWA,
    'GLS': GLS,
    'GradDrop': SafeGradDrop if GradDrop is not None else None,
    'PCGrad': SafePCGrad if PCGrad is not None else None,
    'GradVac': SafeGradVac if GradVac is not None else None,
    'IMTL': SafeIMTL if IMTL is not None else None,
    'CAGrad': SafeCAGrad if CAGrad is not None else None,
    'RLW': RLW,
}

_VALIDATED_STRATEGIES = tuple(name for name, cls in _WEIGHTING_REGISTRY.items() if cls is not None)


class StaticMultiTaskLoss:
    def __init__(self, config):
        self.det_weight = float(config.loss.det_weight)
        self.seg_weight = float(config.loss.seg_weight)

    def backward(self, losses, **kwargs):
        total = self.det_weight * losses['det_loss'] + self.seg_weight * losses['seg_loss']
        total.backward()
        return {
            'det_loss': self.det_weight,
            'seg_loss': self.seg_weight,
        }


class LibMTLWeightingAdapter:
    def __init__(self, config, model):
        strategy_name = config.limtl.strategy
        weighting_cls = _WEIGHTING_REGISTRY.get(strategy_name)
        if weighting_cls is None:
            raise RuntimeError(
                f'LibMTL weighting strategy {strategy_name!r} is unavailable. '
                f'Available strategies: {list_available_strategies()}.')
        self.strategy_name = strategy_name
        self.weight_args = _build_weight_args(config)
        self.strategy = weighting_cls() if weighting_cls is not LegacyEWLikeWeighting else weighting_cls(config, model)
        self.strategy.task_num = 2
        self.strategy.task_name = ['det_loss', 'seg_loss']
        self.strategy.device = torch.device(config.device)
        self.strategy.rep_grad = (self.strategy_name == 'GradDrop')
        self.strategy.get_share_params = lambda: model.backbone.parameters()
        self.strategy.zero_grad_share_params = model.backbone.zero_grad
        self.strategy.init_param()

    def before_epoch(self, epoch):
        self.strategy.epoch = epoch
        if hasattr(self.strategy, 'train_loss_buffer') and (getattr(self.strategy, 'train_loss_buffer', None) is None or epoch == 0):
            self.strategy.train_loss_buffer = torch.ones((2, 1), device=self.strategy.device).cpu().numpy()

    def after_epoch(self, epoch, average_losses):
        if hasattr(self.strategy, 'train_loss_buffer'):
            buffer = getattr(self.strategy, 'train_loss_buffer', None)
            if buffer is None or getattr(buffer, 'shape', None) != (2, 1):
                self.strategy.train_loss_buffer = torch.ones((2, 1), device=self.strategy.device).cpu().numpy()
            self.strategy.train_loss_buffer[0, 0] = max(float(average_losses['det_loss']), 1e-8)
            self.strategy.train_loss_buffer[1, 0] = max(float(average_losses['seg_loss']), 1e-8)

    def backward(self, losses, **kwargs):
        loss_tensor = torch.stack([losses['det_loss'], losses['seg_loss']])
        model_outputs = kwargs.get('model_outputs')
        if self.strategy.rep_grad:
            if model_outputs is None:
                raise RuntimeError('GradDrop requires model_outputs with shared_rep and rep_tasks.')
            self.strategy.rep = model_outputs.get('shared_rep')
            self.strategy.rep_tasks = model_outputs.get('rep_tasks')
            if self.strategy.rep is None or self.strategy.rep_tasks is None:
                raise RuntimeError('GradDrop requires shared_rep and rep_tasks from model outputs.')
        weights = self.strategy.backward(loss_tensor, **self.weight_args)
        if weights is None:
            return None
        if hasattr(weights, 'tolist'):
            weights = weights.tolist()
        return {
            'det_loss': float(weights[0]),
            'seg_loss': float(weights[1]),
        }



def list_available_strategies():
    return list(_VALIDATED_STRATEGIES)



def _build_weight_args(config):
    if config.limtl.strategy == 'GradNorm':
        return {'alpha': 1.5}
    if config.limtl.strategy == 'MGDA':
        return {'mgda_gn': 'none'}
    if config.limtl.strategy == 'DWA':
        return {'T': 2.0}
    if config.limtl.strategy == 'CAGrad':
        return {'calpha': 0.5, 'rescale': 1}
    if config.limtl.strategy == 'GradVac':
        return {'beta': 0.5}
    if config.limtl.strategy == 'GradDrop':
        leak = getattr(getattr(config.limtl, 'graddrop', None), 'leak', 0.0)
        return {'leak': float(leak)}
    if config.limtl.strategy == 'RLW':
        return {'dist': 'Normal'}
    return {}



def create_multitask_weighting(config, model):
    if not config.limtl.enabled:
        return StaticMultiTaskLoss(config)
    return LibMTLWeightingAdapter(config, model)
