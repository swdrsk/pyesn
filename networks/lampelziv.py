# Lempel-Ziv-Welch compression algorithm

def compress(uncompressed):
  """ compress a string to a list of output symbols """
  # build the dictionary
  dict_size = 256
  dictionary = {}
  for i in xrange(dict_size):
    dictionary[chr(i)] = i
 
  result = []

  w = ""
  for c in uncompressed:
    wc = w + c
    if wc in dictionary:
      w = wc
    else:
      result.append(dictionary[w])
      # add wc to the dictionary
      dictionary[wc] = dict_size
      dict_size += 1
      w = c

  # output the code for w
  if w:
    result.append(dictionary[w])

  return result

def decompress(compressed):
  """ decompress a list of output symbols to a string """

  # build the dictionary
  dict_size = 256
  dictionary = {}
  for i in xrange(dict_size):
    dictionary[i] = chr(i)

  if compressed:
    w = chr(compressed.pop(0))
  else:
    raise ValueError, "empty"

  result = [w]

  for k in compressed:
    if k in dictionary:
      entry = dictionary[k]
    elif k == len(dictionary):
      entry = w + w[0]
    else:
      raise ValueError, "Bad compressed k: %d" % k

    result.append(entry)
 
    # add (w + entry.[0]) to the dictionary
    dictionary[dict_size] = w + entry[0]
    dict_size += 1
 
    w = entry

  return result
