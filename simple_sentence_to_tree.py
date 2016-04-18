from nltk import tokenize
initial_sentence = "Hello. Ivan is the coolest name ever (two in this group). How to we create a tree from this sentence."
parts_of_the_sentence = tokenize.sent_tokenize(initial_sentence)
print(parts_of_the_sentence)


words_in_the_part_of_the_sentence = tokenize.word_tokenize(parts_of_the_sentence[2])
print(words_in_the_part_of_the_sentence)


from nltk import tag
tagged_part_of_sentence_with_corresponding_syntactical_value = tag.pos_tag(words_in_the_part_of_the_sentence)
print(tagged_part_of_sentence_with_corresponding_syntactical_value)


from nltk import chunk
tree = chunk.ne_chunk(tagged_part_of_sentence_with_corresponding_syntactical_value)
print(tree)
tree.draw()
