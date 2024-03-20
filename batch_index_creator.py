import utils as ut
import index as ind

if __name__ == "__main__":
    # ut.get_sift()
    # ut.get_glove(200)
    xt, xb, xq, gt = ut.get_sift()

    index_driver = ind.Index("sift", f'IVF{4096},Flat', xb, xb, xq, gt)
    index_driver = ind.Index("sift", f'IVF{2048},Flat', xb, xb, xq, gt)
    index_driver = ind.Index("glove", f'IVF{2048},Flat', xb, xb, xq, gt)
    print(index_driver)