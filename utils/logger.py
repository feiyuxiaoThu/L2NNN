import logging
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from utils.heatmap import *


class TensorBoardLogger(SummaryWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_heatmap(self, tag, matrix, y_title, x_title, epoch, cbarlabel=None, cmap="YlGn"):
        if cbarlabel == None:
            cbarlabel = tag
        fig, ax = plt.subplots()
        im, cbar = heatmap(matrix, y_title, x_title, ax=ax, cmap=cmap, cbarlabel=cbarlabel)
        texts = annotate_heatmap(im, valfmt="{x:.3f}")
        self.add_figure(tag, fig, epoch)

    def add_model_weight_hist(self, model, epoch):
        for name, p in model.named_parameters():
            name = name.replace('.', '/')
            self.add_histogram(tag="weights/{}".format(name),
                               values=p.data.detach().cpu().numpy(),
                               global_step=epoch)

    def add_model_grad_hist(self, model, epoch):
        for name, p in model.named_parameters():
            name = name.replace('.', '/')
            self.add_histogram(tag="grads/{}".format(name),
                               values=p.grad.detach().cpu().numpy(),
                               global_step=epoch)


def get_logger(path, name='log', save=True, show=True):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if save:
        file_handler = logging.FileHandler(path, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.addFilter(logging.Filter(name=logger.name))
        logger.addHandler(file_handler)
    if show:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    return logger


if __name__ == '__main__':
    logger = get_logger('/tmp/log')
    logger.info('info')
    logger.debug('debug')
