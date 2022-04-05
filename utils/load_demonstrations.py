import minerl


def load_demonstrations():
    minerl.data.download('demonstrations', environment='MineRLTreechop-v0')
    minerl.data.download('demonstrations', environment='MineRLObtainDiamondDense-v0')
    minerl.data.download('demonstrations', environment='MineRLObtainIronPickaxeDense-v0')


if __name__ == "__main__":
    load_demonstrations()
