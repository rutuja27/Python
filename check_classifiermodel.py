import numpy as np
import csv
import utils as ut
import sys
from scipy.io import loadmat
import h5py as h5

def match_classifiers(cls_model_matlab, cls_model_biasjaaba, beh_name, nparams, paramsdims):


    cls_model_matlab_rnd = np.round(cls_model_matlab, 6)
    cls_model_biasjaaba_rnd = np.round(cls_model_biasjaaba, 6)
    print('For beh name', beh_name)
    for param_id in range(0,nparams):
        for paramdim_id in range(0,paramsdims):
            if(cls_model_matlab_rnd[param_id][paramdim_id] == cls_model_biasjaaba_rnd[param_id][paramdim_id]):
                continue;
            else:
                print('Matlab model - BIAS model: param id-{:d} param dim-{:d}'
                      ' param_val matlab-{:f} param value bias-{:f}'
                        .format(param_id, paramdim_id, cls_model_matlab[param_id][paramdim_id],
                                cls_model_biasjaaba[param_id][paramdim_id]))


def match_model_params(cls_model_matlab_file, cls_model_biasjaaba_file, beh_names, model_params, cls_dims):

    print('Behviors', beh_names)
    print('Model params', model_params)
    print('Matlab classifier model', cls_model_matlab_file)

    numBehs = len(beh_names)
    nparams = len(model_params)
    paramsdims = cls_dims

    cls_model_biasjaaba = np.zeros((nparams,paramsdims))
    cls_model_matlab = np.zeros((nparams, paramsdims))
    print(cls_model_biasjaaba.shape)

    cls_model_matlab_struct = h5.File(cls_model_matlab_file, 'r')
    for beh_id in range(0,numBehs):
        for param_id in range(0,nparams):
            cls_model_matlab[param_id] = np.array(cls_model_matlab_struct[beh_names[beh_id]][model_params[param_id]]).flatten()
        biasjaaba_file = cls_model_biasjaaba_file + str(beh_id) + '.csv'
        ut.read_classifierparams(biasjaaba_file, cls_model_biasjaaba, paramsdims)
        #print(cls_model_matlab)
        #print(cls_model_biasjaaba)
        match_classifiers(cls_model_matlab, cls_model_biasjaaba, beh_names[beh_id],nparams, cls_dims)

def main():
    print('Number of arguments', len(sys.argv))
    print('Argument list', str(sys.argv))

    if (len(sys.argv) < 5):
        print('Insufficient arguments')
        print('Argument Options:\n' +
              '-classifier model matlab file\n' +
              '-clssifier model biasjaaba online file\n' +
              '-behavior_names\n' +
              '-model_params\n' +
              '-classifier_dims\n'
              )
    else:
        classifier_model_matlab_file = sys.argv[1]
        classifier_model_biasjaaba_file = sys.argv[2]
        behavior_names = sys.argv[3]
        model_params = sys.argv[4]
        classifier_dims = np.int(sys.argv[5])

        behavior_names = [x for x in behavior_names.split(',')]
        model_params = [x for x in model_params.split(',')]
        match_model_params(classifier_model_matlab_file, classifier_model_biasjaaba_file, behavior_names,
                           model_params, classifier_dims)

if __name__ == "__main__":
    main()