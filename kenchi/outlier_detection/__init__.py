from .angle_based import FastABOD
from .classification_based import OCSVM
from .clustering_based import MiniBatchKMeans
from .density_based import LOF
from .distance_based import KNN, OneTimeSampling
from .ensemble import IForest
from .reconstruction_based import PCA
from .statistical import GMM, HBOS, KDE, SparseStructureLearning

DETECTORS = dict(
    fast_abod=FastABOD(),
    oc_svm=OCSVM(),
    mini_batch_k_means=MiniBatchKMeans(),
    lof=LOF(),
    knn=KNN(),
    one_time_sampling=OneTimeSampling(),
    i_forest=IForest(),
    pca=PCA(),
    gmm=GMM(),
    hbos=HBOS(),
    kde=KDE(),
    sparse_structure_learning=SparseStructureLearning(),
)
