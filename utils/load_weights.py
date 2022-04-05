import wandb

load_to = 'train/forger++'
name = 'tviskaron/ForgER/forger:v0'

run = wandb.init(anonymous='allow')
artifact = run.use_artifact(name, type='weights')
artifact_dir = artifact.download(root=load_to)
wandb.finish()
