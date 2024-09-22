import torch.nn as nn
import numpy as np
import torch.utils as utils
import data, network, torch, pickle, time, os
from absl import app,flags

flags.DEFINE_integer('num_steps', int(50000), help='Number of steps of training.')
flags.DEFINE_string('AF', 'PReLUMulti' , help='Choice of activation function.')
flags.DEFINE_string('device', 'GPU' , help='Choice of CPU or GPU.')
flags.DEFINE_string('lossFn','MSE', help=' MSE, L1, or Smooth L1.')
flags.DEFINE_integer('nMLP', int(3), help='Number of MLP layers.')
flags.DEFINE_integer('nConv', int(5), help='Number of Conv layers.')
flags.DEFINE_integer('nGAT', int(5), help='Number of GAT layers.')
flags.DEFINE_integer('HLD', int(128), help='Hidden layer dimension')
flags.DEFINE_integer('batch_size', int(512), help='Training batch size.')
flags.DEFINE_float('lr', 1e-3, help='Learning rate.')
flags.DEFINE_integer('K', int(5), help='Chebconv K.')
flags.DEFINE_integer('nPrintOut', int(100), help='Number of iterations per validation')
flags.DEFINE_integer('rD', int(3), help='Number of input steps.')
flags.DEFINE_bool('targetDx', True, help='Use dx as target or not.')
flags.DEFINE_bool('normDx', True, help='Normalize dx or not.')
flags.DEFINE_bool('normEdge', True, help='Normalize edge for GAT or not.')
flags.DEFINE_bool('newLoss', True, help='To use PCLoss or not')
flags.DEFINE_bool('saveModel', True, help='Save model or not.')
flags.DEFINE_bool('shuffleBatch', True, help='Save model or not.')
flags.DEFINE_string('model', "Hybrid", help="Choose network.")
flags.DEFINE_bool('BN', True, help='To use batch norm or not.')
FLAGS = flags.FLAGS

def main(_):
    if FLAGS.device == 'CPU':
        device = torch.device("cpu")
    if FLAGS.device == 'GPU':
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
            print ("GPU not available, using CPU instead.")
    if FLAGS.lossFn == "MSE":
        lossFn = nn.MSELoss()
    elif FLAGS.lossFn == "SL1":
        lossFn = nn.SmoothL1Loss()
    elif FLAGS.lossFn == "L1":
        lossFn = nn.L1Loss()
    else:
        print ("Loss Func not available, using MSE loss.")
        lossFn = nn.MSELoss()

    #----------------------------- Topology -----------------------------#
    nNodes = 35 
    nVar, nRef = 7, 5
    nodeIdx = np.arange(0,nNodes)
    nodeType = np.array([1, 0, 0, 1/3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1/3, 0, 1, 1/3, 1, 0, 1, 0, 0, 1/3, 0, 1, 0, 1, 1/3, 1, 0, 1, 0, 1/3, 2/3, 2/3]) # 0=empty node, 0.333=constant load, 0.667=PQ DER, 1.0=Droop DER
    stateIdx = network.computeStateIdx(FLAGS.rD,nVar,nRef)
    droopIdx = nodeType==1.0
    DERIdx = nodeType>0.5
    #----------------------------- Set up network parameters -----------------------------#
    nMLP = FLAGS.nMLP
    nConv = FLAGS.nConv
    nGAT = FLAGS.nGAT
    batchSize = FLAGS.batch_size
    iDim = (nVar+nRef)*FLAGS.rD
    oDim = nVar

    modelPara = {
        'nMLP':     nMLP,
        'nConv':    nConv,
        'nGAT':     nGAT,
        'iDim':     iDim,
        'oDim':     oDim,
        'BN':       FLAGS.BN,
        'HLD':      FLAGS.HLD,
        'AF':       FLAGS.AF,
        'K':        FLAGS.K,
        'model':    FLAGS.model,
        'newLoss':  FLAGS.newLoss,
    }

    paraDict = {
    'targetDx':     FLAGS.targetDx,
    'normDx':       FLAGS.normDx,
    'normEdge':     FLAGS.normEdge,
    'rD':           FLAGS.rD,
    'device':       device,
    'nVar':         nVar,
    'nRef':         nRef,
    }

    topoDict = {
        'nodeIdx':      nodeIdx,
        'nodeType':     nodeType,
    }

    modelName = network.modelName(paraDict,topoDict,modelPara)
    model = network.modelSelect(modelPara).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    #opt = torch.optim.AdamW(model.parameters(), lr=FLAGS.lr)
    #opt = torch.optim.SGD(model.parameters(), lr=FLAGS.lr, momentum=0.9)
    
    modelPath = './trainedModel/Full_'+modelName+'.pt'
    metaPath = './trainedModel/Full_'+modelName+'_metadata.pkl'
    checkpointPath = './checkpoint.pt'

    # Load existing metadata and model parameters if any
    if os.path.exists(checkpointPath) and os.path.exists(metaPath):
        with open(metaPath,'rb') as f:
            metadata = pickle.load(f)
            checkpoint = torch.load(checkpointPath,map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            opt.load_state_dict(checkpoint['opt_state_dict'])
            lossBest = checkpoint['lossBest']
            EPOCH = checkpoint['epoch']
        print ("Loading from existing model parameters as initialization.")
        with open('log','a') as f:
            print ("Loading from existing model parameters as initialization.",file=f)
            print ("Model generated",file=f)
            print (model,file=f)
            print (modelPath,file=f)
    else:
        metadata = None
        lossBest = 100
        EPOCH = 0
        print ("No checkpoint found, starting with new model.")
        with open('log','w') as f:
            print ("No checkpoint found, starting with new model.",file=f)
            print ("Model generated",file=f)
            print (model,file=f)
            print (modelPath,file=f)

    #----------------------------- Load all datasets -----------------------------#
    caseList = [f'data{n}' for n in range(21,30)]
    trainIdx = list(np.arange(0,len(caseList),1))
    validIdx = [5]
    for idx in validIdx:
        trainIdx.pop(idx)
    FTG = data.FTGenerator(paraDict,topoDict,caseList,trainIdx,metadata)
    Dataset, metadata = FTG.createDataset()
    print ("Dataset generated.")
    #------------------------- Generate training dataset -----------------------------#
    trainSet = []
    for idx in trainIdx:
        trainSet += Dataset[idx]
    validSet = []
    for idx in validIdx:
        validSet += Dataset[idx]
    # # Remove zero dx data
    # zeroDxNormed = (np.zeros(nVar)-metadata['dxMin'])/(metadata['dxMax']-metadata['dxMin'])
    # for _ss in trainSet:
    #     if torch.sum(_ss.y[0]-torch.DoubleTensor(zeroDxNormed).to(device)) < 1e-4: trainSet.remove(_ss)
    print(f"Training dataset size: {len(trainSet)}")
    with open('log','a') as f:
        print (f"Training dataset size: {len(trainSet)}",file=f)
        print (metadata,file=f)
    #----------------------------- Train the network -----------------------------#
    meta = {
        'varMin': torch.DoubleTensor(metadata['varMin'][:6]),
        'varMax': torch.DoubleTensor(metadata['varMax'][:6]),
        'dxMin':    torch.DoubleTensor(metadata['dxMin'][:6]),
        'dxMax':    torch.DoubleTensor(metadata['dxMax'][:6]),
        'dFreq':    torch.DoubleTensor([metadata['dxMin'][-1],metadata['dxMax'][-1]]),
        'dxNormFactor': metadata['dxNormFactor'],
        'stateIdx':     stateIdx,
        'droopIdx':     droopIdx,
        'DERIdx':       DERIdx,
        'nNodes':       nNodes,
    }

    nEpoch = FLAGS.num_steps
    history = torch.zeros(nEpoch)
    timeStart = time.time()
    nPrintOut = FLAGS.nPrintOut
    if FLAGS.saveModel:
        with open(metaPath, 'wb') as f: pickle.dump(metadata, f)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt,gamma=0.9998)
    for epoch in range(EPOCH,nEpoch):
        loss = network.trainDGNN(model=model,trainSet=list(trainSet),batchSize=batchSize,opt=opt,epoch=epoch+1,scheduler=scheduler,nPrintOut=nPrintOut,paraDict=paraDict,metadata=meta,shuffle=FLAGS.shuffleBatch,lossFn=lossFn)
        history[epoch] = loss
        timer = time.time() - timeStart
        if (epoch+1)%nPrintOut == 0:
            ## Perform validation
            validLoss = network.oneStepEval(model=model,dataset=validSet,batchSize=FLAGS.batch_size,paraDict=paraDict,metadata=meta,shuffle=FLAGS.shuffleBatch,lossFn=lossFn)                    
            print ("LR =",scheduler.get_last_lr())
            print("Iter {2} training loss: {0}, validation loss: {3}, time:{1:1.2f}".format(loss,timer,epoch+1,validLoss))
            print ("")
            with open('log','a') as f:
                print ("LR =",scheduler.get_last_lr(),file=f)
                print("Iter {2} training loss: {0}, validation loss: {3}, time:{1:1.2f}".format(loss,timer,epoch+1,validLoss),file=f)
                print ("",file=f)
            torch.save({
                        'epoch':            epoch,
                        'model_state_dict': model.state_dict(),
                        'opt_state_dict':   opt.state_dict(),
                        'lossBest':         lossBest,
                        }, checkpointPath)
            if validLoss < lossBest:
                lossBest = validLoss
                if FLAGS.saveModel:
                    torch.save({
                        'epoch':            epoch,
                        'model_state_dict': model.state_dict(),
                        'opt_state_dict':   opt.state_dict(),
                        'lossBest':         lossBest,
                        }, checkpointPath)
                    torch.save({
                        'epoch':            epoch,
                        'model_state_dict': model.state_dict(),
                        'opt_state_dict':   opt.state_dict(),
                        }, modelPath)
                    print ("Model saved, loss: {0:1.7e}".format(validLoss))
                    with open('log','a') as f:
                        print ("Model saved, loss: {0:1.7e}".format(validLoss),file=f)

if __name__ == '__main__':
  app.run(main)
