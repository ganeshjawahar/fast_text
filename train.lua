--[[

FastText - Code for training a new model

]]--

require 'torch'
require 'io'
require 'nn'
require 'sys'
require 'os'
require 'xlua'
require 'lfs'
require 'pl.stringx'
require 'pl.file'
tds = require('tds')
utils = require('utils')

cmd = torch.CmdLine()
cmd:option('-input', 'data/ag_news.train', 'training file path')
cmd:option('-output', 'ag_news.t7', 'output file path')
cmd:option('-lr', 0.05, 'learning rate')
cmd:option('-lrUpdateRate', 100, 'change the rate of updates for the learning rate')
cmd:option('-dim', 10, 'size of word vectors')
cmd:option('-epoch', 5, 'number of epochs')
cmd:option('-wordNgrams', 1, 'max length of word ngram')
cmd:option('-seed', 123, 'seed for the randum number generator')
cmd:option('-gpu', 1, 'whether to use gpu (1 = use gpu, 0 = not)')
cmd:option('-preTrain', 0, 'initialize word embeddings with pre-trained vectors?')
cmd:option('-preTrainFile', '../mem_absa/data/glove.6B.300d.txt', 'file containing the pre-trained word embeddings (should be in http://nlp.stanford.edu/projects/glove/ format). this is valid iff preTrain=1.')
params = cmd:parse(arg)
torch.manualSeed(params.seed)

print('building the vocab...')
utils.build_vocab(params)

print('converting the data to tensors...')
utils.get_tensors(params)

print('defining the model...')
params.ngram_lookup = nn.LookupTable(#params.index2ngram, params.dim)
params.linear_pred_layer = nn.Linear(params.dim, #params.index2label, false) -- set 'false' to exclude the bias
params.model = nn.Sequential():add(params.ngram_lookup):add(nn.Mean()):add(params.linear_pred_layer)
params.criterion = nn.CrossEntropyCriterion()

params.ngram_lookup.weight:uniform(-1.0 / params.dim, 1 / params.dim)
params.linear_pred_layer.weight:zero()

if params.preTrain > 0 then
  print('initializing the pre-trained embeddings...')
  local is_present = lfs.attributes(params.preTrainFile) or -1
  if is_present ~= -1 then
    utils.init_word_weights(params)
  else
    print('>>>WARNING>>> Specified pre-trained word embedding file is not found at: ' .. params.preTrainFile)
  end
end

if params.gpu > 0 then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(1)
  params.model = params.model:cuda()
  params.criterion = params.criterion:cuda()
end

print('training the model...')
params.model:training()
params.token_count, params.local_token_count, params.total_progress = 0, 0, (params.epoch * params.ngram_count)
for e = 1, params.epoch do
  xlua.progress(1, #params.dataset)
  local indices, epoch_loss = torch.randperm(#params.dataset), 0
  for i = 1, #params.dataset do
    local input_tensor, label = params.dataset[indices[i]][1], params.dataset[indices[i]][2]
    if params.gpu > 0 then input_tensor = input_tensor:cuda() end
    params.local_token_count = params.local_token_count + (#input_tensor)[1]
    local predictions = params.model:forward(input_tensor)
    local loss = params.criterion:forward(predictions, label)
    epoch_loss = epoch_loss + loss
    local obj_grad = params.criterion:backward(predictions, label)
    params.model:zeroGradParameters()
    params.model:backward(input_tensor, obj_grad)
    params.model:updateParameters(params.lr * (1.0 - (params.token_count / params.total_progress)))
    if params.local_token_count > params.lrUpdateRate then
      params.token_count = params.token_count + params.local_token_count
      params.local_token_count = 0
    end
    if i % 15 == 0 then xlua.progress(i, #params.dataset) end
    if i % 10000 == 0 then collectgarbage() end
  end
  xlua.progress(#params.dataset, #params.dataset)
  print('epoch ' .. e .. ' completed. loss is ' .. (epoch_loss / #params.dataset))
end

print('saving the model...')
torch.save(params.output, params)
