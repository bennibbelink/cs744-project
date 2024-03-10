import utils
import index
def main():

    utils.get_sift()
    xt = utils.fvecs_read("sift/sift_learn.fvecs")
    ind = index.Index("IVF4096,Flat", xt)


if __name__ == "__main__":
    main()