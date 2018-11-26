from .cityscapes import CityscapesDataloader

import torch.utils.data as data

datasets = {
	'cityscapes': CityscapesDataloader,
}


def get_dataset(opt, **kwargs):
    return datasets[opt.dataset](opt, **kwargs)

def get_data_loader(opt, **kwargs):

    dataset = get_dataset(opt, **kwargs)

    training = kwargs['training']
    data_loader = data.DataLoader(dataset,
                                  batch_size=opt.batch_size,
                                  num_workers=opt.num_workers,
                                  shuffle=training,
                                  pin_memory=True)
    return data_loader
