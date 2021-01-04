from preprocessing.multiP_examplar import ExamplarCluster
from preprocessing.multiP_Kmeans import KmeansCluster
from preprocessing.multiP_similarity import Similarity, SimilarityShapenet
if __name__ == '__main__':
    # EC = ExamplarCluster(process_num=50, C=4, subset=True)
    # EC = ExamplarCluster(process_num=50, C=1, subset=False, test_mode=True, normalize=False, negative_weight_rescale=1.0)
    # EC.multi_process_generate()

    # EC = KmeansCluster(process_num=40, K=4, subset=False, test_mode=True)
    # EC.multi_process_generate()

    # SI = Similarity(process_num=10, save_path='/local/vondrick/cz/ImageNet-Data/ResNet152features_train_overlap_obj/similarity',
    #                 input_path='/local/vondrick/cz/ImageNet-Data/ResNet152features_train_overlap_obj/train')
    # SI.multi_process_generate()

    # SIS = SimilarityShapenet(process_num=1, save_path='/proj/vondrick/mcz/ShapeNet/Similarity/train')
    # SIS.multi_process_generate()


    # TODO: Done! here Gnerate training subset of imagenet that are overlapping with objectnet categories
    from preprocessing.obj_img_nonoverlap_id import gen_imagenet_overlap_data
    # gen_imagenet_overlap_data("/local/vondrick/cz/ImageNet-Data/train", "/local/vondrick/cz/ImageNet-Data/train_overlap_obj")
    gen_imagenet_overlap_data("/local/vondrick/cz/ImageNet-Data/val", "/local/vondrick/cz/ImageNet-Data/val_overlap_obj")