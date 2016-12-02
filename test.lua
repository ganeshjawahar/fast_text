--[[

FastText - Code for testing a trained model

]]--

require 'torch'
require 'io'
require 'nn'
require 'os'
require 'xlua'
require 'pl.stringx'
require 'pl.file'
tds = require('tds')
utils = require('utils')

cmd = torch.CmdLine()
cmd:option('-model', 'ag_news.t7', 'trained model file path')
cmd:option('-test', 'data/ag_news.test', 'testing file path')
cmd:option('-gpu', 1, 'whether to use gpu (1 = use gpu, 0 = not). use the same option used to train the model.')
test_params = cmd:parse(arg)

if test_params.gpu > 0 then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(1)
end

print('loading the trained model...')
train_params = torch.load(test_params.model)

print('testing for the new data...')
train_params.model:evaluate()
local acc, total = 0, 0
for line in io.lines(test_params.test) do
  local label, text = utils.get_tuple(line)
  local ngrams = utils.tokenize(text, train_params.wordNgrams)
  local chooped_ngrams = utils.chop_ngram(train_params, ngrams)
  if #chooped_ngrams > 0 then
    local tensor = torch.Tensor(chooped_ngrams)
    if train_params.gpu > 0 then tensor = tensor:cuda() end
    local predictions = train_params.model:forward(tensor)
    local _, max_ids = predictions:max(1)
    if max_ids[1] == train_params.label2index[label] then
      acc = acc + 1
    end
  end
  total = total + 1
end
print('accuracy on test data [' .. test_params.test .. '] = ' .. (acc / total))
