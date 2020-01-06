from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from argparse import ArgumentParser

from vector_search import train_model
from vector_search import vector_search
from utils import load_paired_img_wrd

import os
def build_parser():
    par = ArgumentParser()
    par.add_argument('--glove_path', type=str,
                     dest='glove_path', help='path to pre-trained GloVe vectors', required=True)
    par.add_argument('--dataset_path', type=str,
                     dest='dataset_path', help='path to dataset', required=True)
    return par


if __name__ == "__main__":
    # parser = build_parser()
    # options = parser.parse_args()
    # glove_path = options.glove_path
    # dataset_path = options.dataset_path
    # word_vectors = vector_search.load_glove_vectors(glove_path)
    # images, vectors, image_paths = load_paired_img_wrd(dataset_path, word_vectors)

    os.system("python ./vector_search/train_model.py --model_name resnet --dataset_path ../Datasets/images --num_epochs 500 --batch_size 8 --feature_extract True")