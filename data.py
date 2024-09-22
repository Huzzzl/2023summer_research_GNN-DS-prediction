import numpy as np
from scipy.io import loadmat
import torch as torch
from torch_geometric import utils
from torch_geometric.data import Data
from torch.utils.data import Dataset

path = '../../DroopData/'
dxNormFactor = 1.
uFactor = 1e1
K = 2.

## Utility functions
def MinMaxNorm(X,min,max):
    return (X-min)/(max-min)

def convertToComplexValue(x):
    # # Convert Vmag, Vphase, Imag, and Iphase to complex number
    Vc = x[:,:,0] * np.exp(1j*x[:,:,1])
    x[:,:,0], x[:,:,1] = np.real(Vc), np.imag(Vc)
    Ic = x[:,:,2] * np.exp(1j*x[:,:,3])
    x[:,:,2], x[:,:,3] = np.real(Ic), np.imag(Ic)
    return x

def getEdge(caseName,normEdge):
    _admittanceMatrix = loadmat(f'{path}{caseName}/groupedData_processed.mat')['Y'].todense()
    _mask = np.where(_admittanceMatrix!=0)
    _admittanceMatrix[_mask] = 1./_admittanceMatrix[_mask]
    _admittanceMatrix -= np.diag(np.diag(_admittanceMatrix)) # Remove self connections
    _Normed = np.abs(_admittanceMatrix)
    _Normed = np.exp(-K*_Normed)
    _Normed[_Normed==1] = 0.
    _edgeIdx, _edgeAttr = utils.dense_to_sparse(torch.FloatTensor(_Normed))
    return _edgeIdx,_edgeAttr

class FTGenerator(object):
    def __init__(self, parameterDict, topologyDict, caseList, trainIdx, metadata=None):
        self.device =           parameterDict['device']
        self.rD =               parameterDict['rD']
        self.targetDx =         parameterDict['targetDx']   
        self.normDx =           parameterDict['normDx']                    
        self.normEdge =         parameterDict['normEdge']
        self.nVar =             parameterDict['nVar']
        self.nRef =             parameterDict['nRef']

        self.nodeType =         np.array(topologyDict['nodeType'])
        self.nodeIdx =          topologyDict['nodeIdx']
        self.nNodes =           len(self.nodeIdx)
        self.emptyNodeIdx =     np.where(self.nodeType==0)[0]
        self.caseList =         caseList
        self.nCase =            len(caseList)
        self.trainIdx =         trainIdx
        
        assert self.rD>=1,'recurrentDim cannot be < 1.'
        if metadata is None:
            self.__extractMetadata()
        else:
            self.metadata = metadata
        assert self.metadata is not None

    def __extractMetadata(self):
        self.metadata = {}
        _rawStateList = [self.__constructRawData(self.caseList[_idx])[:,:,:self.nVar] for _idx in self.trainIdx]
        _refFreqList = [self.__constructRawData(self.caseList[_idx])[:,:,-3] for _idx in self.trainIdx]
        _minState = [np.min(_list.reshape(-1,self.nVar),axis=0) for _list in _rawStateList]
        _minState = np.array(np.min(_minState,axis=0))
        _maxState = [np.max(_list.reshape(-1,self.nVar),axis=0) for _list in _rawStateList]
        _maxState = np.array(np.max(_maxState,axis=0))
        _minFreqRef = np.min([np.min(_list) for _list in _refFreqList])
        _maxFreqRef = np.max([np.max(_list) for _list in _refFreqList])
        _normStateList = [MinMaxNorm(_list,_minState,_maxState) for _list in _rawStateList]
        _dxList = [_list[1:]-_list[:-1] for _list in _normStateList]
        # Construct metadata with state+ref values
        _minRef = np.array([_minState[4],_minState[5],_minFreqRef, 0., 0.])
        _maxRef = np.array([_maxState[4],_maxState[5],_maxFreqRef, 1., 1.])
        _minState = np.hstack([_minState,_minRef])
        _maxState = np.hstack([_maxState,_maxRef])
        self.metadata['varMin'] = _minState
        self.metadata['varMax'] = _maxState
        # Extract metadata for dx
        _dxMin = [np.min(_list.reshape(-1,self.nVar),axis=0) for _list in _dxList]
        self.metadata['dxMin'] = np.array(np.min(_dxMin,axis=0))
        _dxMax = [np.max(_list.reshape(-1,self.nVar),axis=0) for _list in _dxList]
        self.metadata['dxMax'] = np.array(np.max(_dxMax,axis=0))
        self.metadata['dxNormFactor'] = dxNormFactor
    
    def __constructRawData(self,caseName):
        # Construct a complete matrix that includes [P,Q,V,delta,I,theta,Pref,Qref,nodeType,disturbance]
        _data = loadmat(f'{path}{caseName}/groupedData_processed.mat')
        _disturbPos = loadmat(f'{path}{caseName}/disturbance_position.mat')['disturbance_position'].squeeze()

        _nSteps = _data['t'].shape[1]
        _nNodes = len(_data['bus_data'])
        _nLoads = len(_data['load_data'])
        _nGen = len(_data['gen_data'])
        
        _raw = np.zeros([_nSteps,_nNodes,self.nVar+self.nRef])

        # Frequency
        _genFreq = []
        for _n in range(_nGen):
            _tp = _data['gen_data'][_n][0]
            _genFreq.append(_tp[2].squeeze())
        _genFreq = np.array(_genFreq).T
        _genIdx = self.nodeType>0.5
        _droopIdx = self.nodeType==1.0
        _raw[:,_genIdx,self.nVar-1] = _genFreq
        _freqRef = np.mean(_raw[:,_droopIdx,self.nVar-1],axis=1)
        _raw[:,~_genIdx,self.nVar-1] = np.tile(_freqRef,(len(self.nodeType)-_nGen,1)).T

        for _n in range(_nNodes):
            _tp = _data['bus_data'][_n][0]
            # [V,delta,I,theta,P,Q]
            _raw[:,_n,:self.nVar-1] = np.array([_tp[_i].squeeze() for _i in range(self.nVar-1)]).T
            # # Ref frequency
            _raw[:,_n,-3] = _freqRef
            # [Disturbance location]
            _raw[:,_n,-2] = _disturbPos
            # [NodeType]
            _raw[:,_n,-1] = self.nodeType[_n]

        # [Pref, Qref] for loads
        for _n in range(_nLoads):
            _tp = _data['load_data'][_n][0]
            _idx = int(_tp[-1].todense())-1
            _raw[:,_idx,self.nVar:self.nVar+2] = np.array([_tp[_i].squeeze() for _i in range(2)]).T
        # [Pref, Qref] for DERs
        for _n in range(_nGen):
            _tp = _data['gen_data'][_n][0]
            _idx = int(_tp[-1].todense())-1
            _raw[:,_idx,self.nVar:self.nVar+2] = np.array([_tp[_i].squeeze() for _i in range(2)]).T
        # Convert the mag&phase in V&I to complex number
        _raw = convertToComplexValue(_raw)

        return _raw

    def __computeDx(self,data):
        _dx = data[1:]-data[:-1]
        if self.normDx:
            _dx = MinMaxNorm(_dx, self.metadata['dxMin'],self.metadata['dxMax'])
        return _dx*self.metadata['dxNormFactor']
    
    def __createFT(self,caseName):
        _rawData = self.__constructRawData(caseName)
        _normData = MinMaxNorm(_rawData,self.metadata['varMin'],self.metadata['varMax'])
        ## Replace PQRef to dPQRef
        _normData[:-1,:,[self.nVar,self.nVar+1]] = (_normData[1:,:,[self.nVar,self.nVar+1]]-_normData[:-1,:,[self.nVar,self.nVar+1]])*uFactor
        # Assemble FT
        _nSteps = _normData.shape[0]-self.rD
        _featureDim = (self.nVar+self.nRef)*self.rD
        _feature = np.zeros([_nSteps,self.nNodes,_featureDim])
        _target = []
        for _n in range(_nSteps):
            _states = _normData[_n:_n+self.rD]
            _states = np.transpose(_states,(1,0,2)).reshape(self.nNodes,-1)
            _feature[_n] = _states
            _target.append(_normData[_n+self.rD,:,:self.nVar])
        
        _target = self.__computeDx(_normData[:,:,:self.nVar])[(self.rD-1):] if self.targetDx else np.array(_target)
        return _feature,_target

    def createDataset(self):
        _Dataset = []
        for _case in self.caseList:
            _dataList = []
            _featureList,_targetList = self.__createFT(_case)
            _edgeIdx, _edgeAttr = getEdge(_case,self.normEdge)
            for _n in range(len(_featureList)):
                _dataList.append(Data(x=torch.DoubleTensor(_featureList[_n]), y=torch.DoubleTensor(_targetList[_n]), edge_index=_edgeIdx, edge_attr=_edgeAttr).to(self.device))
            _Dataset.append(_dataList)
        return _Dataset,self.metadata
