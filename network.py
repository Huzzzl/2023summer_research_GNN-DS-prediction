from torch_geometric.loader import DataLoader as batchLoader
import torch,random
import numpy as np
from torch_geometric.nn import ChebConv, GATv2Conv
import torch.nn as nn
torch.set_printoptions(precision=7)

PCLOSS_FACTOR = 1.

def Norm(x,min,max):
    return (x-min)/(max-min)

def deNorm(x,min,max):
    return x*(max-min)+min

def computeStateIdx(rD,nVar,nRef):
    return np.arange(0,rD*(nVar+nRef))[-(nVar+nRef):-nRef-1]
    
class GNN(torch.nn.Module):

    def __init__(self,modelPara):
        super(GNN, self).__init__()
        self.nConvLayers = modelPara['nConv']
        self.nMLPLayers = modelPara['nMLP']
        self.coe = 2
        if modelPara['BN']:
            self.coe += 1
        _HLD = modelPara['HLD']
        _AF = ifAF(modelPara['AF'],_HLD)
        _encode = []
        _decode = []
        _conv = []
        
        _encode.append(torch.nn.Linear(modelPara['iDim'],_HLD))
        _encode.append(ifAF(modelPara['AF'],_HLD))
        if modelPara['BN']:
            _encode.append(nn.BatchNorm1d(_HLD))
        for _ in range(self.nMLPLayers-1):
            _encode.append(torch.nn.Linear(_HLD,_HLD))
            _encode.append(_AF)
            if modelPara['BN']:
                _encode.append(nn.BatchNorm1d(_HLD))
        for _ in range(self.nConvLayers):
            _conv.append(ChebConv(_HLD,_HLD,K=modelPara['K']))
            _conv.append(_AF)
            if modelPara['BN']:
                _conv.append(nn.BatchNorm1d(_HLD))
        for _ in range(self.nMLPLayers-1):
            _decode.append(torch.nn.Linear(_HLD,_HLD))
            _decode.append(_AF)
            if modelPara['BN']:
                _decode.append(nn.BatchNorm1d(_HLD))
        _decode.append(torch.nn.Linear(_HLD,modelPara['oDim']))
        _decode.append(ifAF(modelPara['AF'],modelPara['oDim']))
        if modelPara['BN']:
            _decode.append(nn.BatchNorm1d(modelPara['oDim']))

        self.encodeLayers = torch.nn.ModuleList(_encode)
        self.convLayers = torch.nn.ModuleList(_conv)
        self.decodeLayers = torch.nn.ModuleList(_decode)
        
    def forward(self, x, edgeIdx, edgeAttr):
        ## Forwarding
        for _f in self.encodeLayers:
            x = _f(x)
        for _n in range(int(len(self.convLayers)/self.coe)):
            x = self.convLayers[self.coe*_n](x, edgeIdx, edgeAttr)
            for _m in range(1,self.coe):
                x = self.convLayers[self.coe*_n+_m](x)
        for _f in self.decodeLayers:
            x = _f(x)
        return x

class Hybrid(torch.nn.Module):

    def __init__(self,modelPara):
        super(Hybrid, self).__init__()
        self.nConvLayers = modelPara['nConv']
        self.nMLPLayers = modelPara['nMLP']
        self.nGATLayers = modelPara['nGAT']
        self.coe = 2
        if modelPara['BN']: self.coe += 1
        _HLD = modelPara['HLD']
        _AF = ifAF(modelPara['AF'],_HLD)
        _encode = []
        _decode = []
        _conv = []
        _gat = []
        
        for _n in range(self.nMLPLayers):
            _inChannel = modelPara['iDim'] if _n==0 else _HLD
            _outChannel = _HLD
            _encode.append(torch.nn.Linear(_inChannel,_outChannel))
            # if modelPara['BN']: _encode.append(nn.BatchNorm1d(_outChannel))
            _encode.append(_AF)
        for _ in range(self.nGATLayers):
            _gat.append(GATv2Conv(_HLD,_HLD,heads=2,concat=False,add_self_loops=False))
            _gat.append(_AF)
            if modelPara['BN']: _gat.append(nn.BatchNorm1d(_HLD))
            
        for _ in range(self.nConvLayers):
            _conv.append(ChebConv(_HLD,_HLD,K=modelPara['K']))
            _conv.append(_AF)
            if modelPara['BN']: _conv.append(nn.BatchNorm1d(_HLD))
            
        for _n in range(self.nMLPLayers):
            _inChannel = _HLD
            _outChannel = modelPara['oDim'] if _n==self.nMLPLayers-1 else _HLD
            _decode.append(torch.nn.Linear(_inChannel,_outChannel))
            # if modelPara['BN']: _decode.append(nn.BatchNorm1d(_outChannel))
            _decode.append(ifAF(modelPara['AF'],_outChannel))

        self.encodeLayers = torch.nn.ModuleList(_encode)
        self.convLayers = torch.nn.ModuleList(_conv)
        self.decodeLayers = torch.nn.ModuleList(_decode)
        self.GATLayers = torch.nn.ModuleList(_gat)

    def forward(self, x, edgeIdx, edgeAttr):
        ## Forwarding
        for _f in self.encodeLayers:
            x = _f(x)
        for _n in range(int(len(self.GATLayers)/self.coe)):
            x, (_,_alpha) = self.GATLayers[self.coe*_n](x, edgeIdx, None, True)
            for _m in range(1,self.coe):
                x = self.GATLayers[self.coe*_n+_m](x)
        _alpha = torch.mean(_alpha,axis=1).reshape(-1,1)
        _newEdgeAttr = torch.cat((_alpha,edgeAttr.reshape(-1,1)),1)
        _newEdgeAttr = torch.mean(_newEdgeAttr,axis=1)
        for _n in range(int(len(self.convLayers)/self.coe)):
            x = self.convLayers[self.coe*_n](x, edgeIdx, _newEdgeAttr)
            for _m in range(1,self.coe):
                x = self.convLayers[self.coe*_n+_m](x)
        for _f in self.decodeLayers:
            x = _f(x)
        return x

class GAT(torch.nn.Module):

    def __init__(self,modelPara):
        super(GAT, self).__init__()
        self.nConvLayers = modelPara['nConv']
        self.nMLPLayers = modelPara['nMLP']
        self.nGATLayers = modelPara['nGAT']
        self.coe = 2
        if modelPara['BN']: self.coe += 1
        _HLD = modelPara['HLD']
        _AF = ifAF(modelPara['AF'],_HLD)
        _encode,_decode,_gat = [],[],[]
        
        for _n in range(self.nMLPLayers):
            _inChannel = modelPara['iDim'] if _n==0 else _HLD
            _outChannel = _HLD
            _encode.append(torch.nn.Linear(_inChannel,_outChannel))
            _encode.append(_AF)
        for _ in range(self.nGATLayers):
            _gat.append(GATv2Conv(_HLD,_HLD,heads=3,concat=False,add_self_loops=False))
            _gat.append(_AF)
            if modelPara['BN']:
                _gat.append(nn.BatchNorm1d(_HLD))
        for _n in range(self.nMLPLayers):
            _inChannel = _HLD
            _outChannel = modelPara['oDim'] if _n==self.nMLPLayers-1 else _HLD
            _decode.append(torch.nn.Linear(_inChannel,_outChannel))
            _decode.append(ifAF(modelPara['AF'],_outChannel))

        self.encodeLayers = torch.nn.ModuleList(_encode)
        self.decodeLayers = torch.nn.ModuleList(_decode)
        self.GATLayers = torch.nn.ModuleList(_gat)

    def forward(self, x, edgeIdx, edgeAttr):
        ## Forwarding
        for _f in self.encodeLayers:
            x = _f(x)
        for _n in range(int(len(self.GATLayers)/self.coe)):
            x = self.GATLayers[self.coe*_n](x, edgeIdx)
            for _m in range(1,self.coe):
                x = self.GATLayers[self.coe*_n+_m](x)
        for _f in self.decodeLayers:
            x = _f(x)
        return x

def modelSelect(modelPara):
    # modelPara should contain all the model parameters.
    # 'HLD' : contains lists of input and output dimension for each layer.
    # 'AF': contains a list of AF.
    #         ** for GNN model, this should also include the AF for the  two linear layers on two ends.
    # 'K'   : contains a list of Chebconv K for each Conv layer. 
    _modelName = modelPara['model'] 
    if _modelName == 'GNN':
        _model = GNN(modelPara).double()
    elif _modelName == "GAT":
        _model = GAT(modelPara).double()
    elif _modelName == "Hybrid":
        _model = Hybrid(modelPara).double()
    else:
        print ("Model not recognized.")
    return _model

def ifAF(AF,HLD):
    if AF == 'ReLU':
        return nn.ReLU()
    elif AF == "LeakyReLU":
        return nn.LeakyReLU()
    elif AF == "SiLU":
        return nn.SiLU()
    elif AF == "CELU":
        return nn.CELU()
    elif AF == 'GELU':
        return nn.GELU()
    elif AF == 'SELU':
        return nn.SELU()
    elif AF == "sigmoid":
        return nn.Sigmoid()
    elif AF == "PReLU":
        return nn.PReLU()
    elif AF == "PReLUMulti":
        return nn.PReLU(HLD)
    elif AF == 'tanh':
        return nn.Tanh()
    else:
        return lambda x: x
        
def modelName(paraDict,topoDict,modelPara):
    _name = modelPara['model']
    if paraDict['targetDx']:
        _name += "_dx"
    if paraDict['normDx']:
        _name += "Norm"
    if modelPara['newLoss']:
        _name += "_newLoss"
    _name += '_'+str(modelPara['nMLP'])+'MLP'+str(modelPara['nConv'])+'Conv_'
    _name += modelPara['AF']
    _name += str(modelPara['HLD'])+'_'
    _name += "K"+str(modelPara['K'])+'_'
    _name += '_RD'+str(paraDict['rD'])
    _name += '_'+str(len(topoDict['nodeIdx']))+'Node'
    return _name

def mainLossFn(yHat,y,fn):
    return fn(yHat,y)

def PCLossFn(x,yHat,paraDict,metadata):
    _device = yHat.device
    # Compute actual states
    _varMin, _varMax =metadata['varMin'].to(_device), metadata['varMax'].to(_device)
    if paraDict['targetDx']:
        if paraDict['normDx']:
            _minDx, _maxDx = metadata['dxMin'].to(_device), metadata['dxMax'].to(_device)
            _dx = deNorm(yHat[:,:6]/metadata['dxNormFactor'],_minDx,_maxDx)
        else:
            _dx = yHat[:,:6]/metadata['dxNormFactor']
        _states = deNorm(x[:,metadata['stateIdx']]+_dx,_varMin,_varMax) # dx->x (new step) then denormalize
    else:
        _states = deNorm(yHat[:,:6],_varMin,_varMax)
    # Compute loss 
    _Pc = _states[:,4]+_states[:,5]*1j
    _Vc = _states[:,0]+_states[:,1]*1j
    _Ic = _states[:,2]+_states[:,3]*1j
    _diff = torch.abs(_Pc-_Vc*torch.conj(_Ic))
    return torch.linalg.norm(_diff)**2

def refFreqLoss(x,yHat,metadata):
    _device = yHat.device
    _min,_max = metadata['dFreq'].to(_device)
    # DeNorm delta freq
    _dFreq = yHat[:,-1]*(_max-_min)+_min
    # Compute actual frequencies         
    _newFreq = (x[:,-6]+_dFreq).reshape(-1,metadata['nNodes'])
    _newFreqRef = torch.mean(_newFreq[:,metadata['droopIdx']],axis=1)
    _FreqRefHat = torch.mean(_newFreq[:,~metadata['DERIdx']],axis=1)
    return torch.linalg.norm(_newFreqRef-_FreqRefHat)**2

def trainDGNN(model,trainSet,batchSize,opt,epoch,scheduler,nPrintOut,paraDict,metadata,shuffle,lossFn=nn.MSELoss()):
    model.train()
    _lossList = []
    _trainLoader = batchLoader(trainSet,batch_size=batchSize,shuffle=shuffle)
    for _batch in _trainLoader:
        opt.zero_grad(set_to_none=True)
        _yHat = model(_batch.x,_batch.edge_index,_batch.edge_attr)
        _loss1 = mainLossFn(_yHat,_batch.y,lossFn)
        _loss2 = PCLossFn(_batch.x,_yHat,paraDict,metadata)*PCLOSS_FACTOR
        _loss3 = refFreqLoss(_batch.x,_yHat,metadata)
        _loss = _loss1+_loss2+_loss3
        _loss.backward(retain_graph=False)
        _lossList.append(_loss.item())
        opt.step()
    scheduler.step()

    if  epoch%nPrintOut==0:
        _diff = torch.abs(_yHat-_batch.y)
        _where = torch.where(_diff==torch.max(_diff))
        _PcLossRef = PCLossFn(_batch.x,_batch.y,paraDict,metadata)
        print ("PcLossRef:",_PcLossRef)
        print ("Training:")
        print (np.min(_lossList),np.max(_lossList))
        print ("Max and min diff:",torch.max(_diff),torch.min(_diff))
        print ("Max diff pred vs truth:")
        print (_yHat[_where[0]])
        print (_batch.y[_where[0]])
        _pick = random.randint(0,len(_yHat)-1)
        print ("Compare:",_pick)
        print (_yHat[_pick])
        print (_batch.y[_pick])
        print (f"Main loss {_loss1:1.7e}, PC loss {_loss2:1.7e}")
        with open('log','a') as f:
            print ("Training:",file=f)
            print (np.min(_lossList),np.max(_lossList),file=f)
            print ("Max and min diff:",torch.max(_diff),torch.min(_diff),file=f)
            print ("Max diff pred vs truth:",file=f)
            print (_yHat[_where[0]],file=f)
            print (_batch.y[_where[0]],file=f)
            print ("Compare:",_pick,file=f)
            print (_yHat[_pick],file=f)
            print (_batch.y[_pick],file=f)
            print (f"Main loss {_loss1:1.7e}, PC loss {_loss2:1.7e}",file=f)
    return np.mean(_lossList)

def oneStepEval(model,dataset,batchSize,paraDict,metadata,shuffle,lossFn=nn.MSELoss()):
    _dataloader = batchLoader(dataset,batch_size=batchSize,shuffle=shuffle)
    model.eval()
    _lossList = []
    for _batch in _dataloader:
        _yHat = model(_batch.x,_batch.edge_index,_batch.edge_attr)
        _loss = mainLossFn(_yHat,_batch.y,lossFn)+PCLossFn(_batch.x,_yHat,paraDict,metadata)*PCLOSS_FACTOR
        _lossList.append(_loss.item())
    _diff = torch.abs(_yHat-_batch.y)
    _where = torch.where(_diff==torch.max(_diff))
    print ("")
    print ("Validation:")
    print ("Max and min diff:",torch.max(_diff),torch.min(_diff))
    print ("Max diff pred vs truth:")
    print (_yHat[_where[0]])
    print (_batch.y[_where[0]])
    _pick = random.randint(0,len(_yHat)-1)
    print ("Compare:",_pick)
    print (_yHat[_pick])
    print (_batch.y[_pick])
    print ("")
    with open('log','a') as f:
        print ("",file=f)
        print ("Validation:",file=f)
        print ("Max and min diff:",torch.max(_diff),torch.min(_diff),file=f)
        print ("Max diff pred vs truth:",file=f)
        print (_yHat[_where[0]],file=f)
        print (_batch.y[_where[0]],file=f)
        print ("Compare:",_pick,file=f)
        print (_yHat[_pick],file=f)
        print (_batch.y[_pick],file=f)
        print ("",file=f)
    return np.mean(_lossList)
