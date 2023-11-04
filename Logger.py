from torch.utils.tensorboard import SummaryWriter

class Logger():
    def __init__(self):
        self.writer = SummaryWriter()

    def add_loss(self, loss_value, gae_type, epoch):
        title = 'train_loss/GAE_{}'.format(gae_type)
        self.writer.add_scalar(title, loss_value, epoch)

    def close(self):
        self.writer.close()