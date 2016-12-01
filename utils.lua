--[[

FastText - Utilities

]]--

local utils = {}

function utils.get_tuple(line)
  local content = stringx.split(line, ',')
  local label, res = stringx.strip(content[1]), ''
  for i = 3, #content do
    res = res .. stringx.strip(content[i]) .. ' '
  end
  return label, stringx.strip(res)
end

function utils.tokenize(text, wordNgrams)
  local ngrams, words = {}, stringx.split(text)
  for i = 1, #words do
    table.insert(ngrams, words[i])
  end
  for size = 2, wordNgrams do
    if #words >= size then
      for start = 1, (#words - size + 1) do
        local ngram = ''
        for i = 1, size do
          ngram = ngram .. words[start + i - 1] .. '$'
        end
        table.insert(ngrams, ngram)
      end
    end
  end
  return ngrams
end

-- Function to build the vocabulary
function utils.build_vocab(params)
  params.ngram2index, params.index2ngram = tds.hash(), tds.hash()
  params.label2index, params.index2label = tds.hash(), tds.hash()
  params.ngram_count = 0
  for line in io.lines(params.input) do
    local label, text = utils.get_tuple(line)
    local ngrams = utils.tokenize(text, params.wordNgrams)
    for i = 1, #ngrams do
      local ngram = ngrams[i]
      if params.ngram2index[ngram] == nil then
        params.index2ngram[#params.index2ngram + 1] = ngram
        params.ngram2index[ngram] = #params.index2ngram
      end
    end
    if params.label2index[label] == nil then
      params.index2label[#params.index2label + 1] = label
      params.label2index[label] = #params.index2label
    end
    params.ngram_count =  params.ngram_count + #ngrams
  end
  print(#params.index2ngram .. ' unique ngrams found\n' .. #params.index2label .. ' unique labels found')
end

-- Function to get the tensors
function utils.get_tensors(params)
  params.dataset = {}
  for line in io.lines(params.input) do
    local label, text = _get_tuple(line)
    local ngrams = _tokenize(text, params.wordNgrams)
    local tensor = torch.Tensor(#ngrams)
    for i = 1, #ngrams do
      tensor[i] = params.ngram2index[ngrams[i]]
    end
    table.insert(params.dataset, {tensor, params.label2index[label]})
  end
  print(#params.dataset .. ' records found')
end

-- Function to initialize the pre-trained word emebeddings
function utils.init_word_weights(params)
  local p = 0
  for line in io.lines(params.preTrainFile) do
    local content = stringx.split(line)
    local word = content[1]
    if params.ngram2index[word] ~= nil and params.dim == (#content - 1) then
      local tensor = torch.Tensor(#content - 1)
      for i = 2, #content do
        params.ngram_lookup.weight[params.ngram2index[word]][i - 1] = tonumber(content[i])
      end
      p = p + 1
    end
  end
  print(p .. ' words initialized')
end

-- Function to chop ngrams
function utils.chop_ngram(params, ngrams)
  local res_ngrams = {}
  for i = 1, #ngrams do
    if params.ngram2index[ngrams[i]] ~= nil then
      table.insert(res_ngrams, params.ngram2index[ngrams[i]])
    end
  end
  return res_ngrams
end

return utils