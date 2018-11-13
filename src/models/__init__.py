from .resnet101_baseline import get_resnet101_baseline
from .resnet101_base_oc import get_resnet101_base_oc_dsn
from .resnet101_pyramid_oc import get_resnet101_pyramid_oc_dsn
from .resnet101_asp_oc import get_resnet101_asp_oc_dsn


models = {
    'baseline': get_resnet101_baseline,
    'base_oc_dsn': get_resnet101_base_oc_dsn,
    'pyramid_oc_dsn': get_resnet101_pyramid_oc_dsn,
    'asp_oc_dsn': get_resnet101_asp_oc_dsn,
}

def get_model(opt, **kwargs):
    return models[opt.model_type](**kwargs)
