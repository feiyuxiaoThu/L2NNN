import visdom
import time
import numpy as np


class Visualizer(object):
    '''
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    调用原生的visdom接口
    '''

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

        # 画的第几个数，相当于横座标
        # 保存（’loss',23） 即loss的第23个点
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        '''
        修改visdom的配置
        '''
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        '''
        一次plot多个
        @params d: dict (name,value) i.e. ('loss',0.11)
        '''
        for k, v in d.iteritems():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.iteritems():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        '''
        Plot with y and implicit index.
        self.plot('loss',1.00)
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def plot2(self, name, x, y, **kwargs):
        '''
        Plot with specified x and y.
        self.plot('loss', 1, 2.0)
        '''
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )

    def img(self, name, img_, **kwargs):
        '''
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)

        ！！！do not ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~！！！
        '''
        self.vis.images(img_.cpu().numpy(),
                        win=unicode(name),
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        '''
        self.log({'loss':1,'lr':0.0001})
        '''

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)


# TSNE visualization

def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


def vis_tsne(X, y):
    proj = TSNE(random_state=0).fit_transform(X)
    scatter(proj, y)
