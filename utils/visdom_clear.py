import visdom

vis = visdom.Visdom(port=22222)
envs = []

for env in vis.get_env_list():
    if not env in envs:
        vis.delete_env(env)
