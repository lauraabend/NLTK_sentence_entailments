from nltk import tokenize

initial_sentence = "Hello. Group A is the coolest group ever. How do we create a tree from this sentence."
parts_of_the_sentence = tokenize.sent_tokenize(initial_sentence)
print(parts_of_the_sentence)
print(parts_of_the_sentence)


words_in_the_part_of_the_sentence = tokenize.word_tokenize(parts_of_the_sentence[2])
print("words_in_the_part_of_the_sentence")
print(words_in_the_part_of_the_sentence)


from nltk import tag
tagged_part_of_sentence_with_corresponding_syntactical_value = tag.pos_tag(words_in_the_part_of_the_sentence)
print("tagged_part_of_sentence_with_corresponding_syntactical_value")
print(tagged_part_of_sentence_with_corresponding_syntactical_value)


from nltk import chunk
from nltk.tree import Tree
tree = chunk.ne_chunk(tagged_part_of_sentence_with_corresponding_syntactical_value)
print("tree")
print(tree)
#tree.draw()


the_tree = Tree("sentence", tokenize.sent_tokenize(initial_sentence))

#the_tree.draw()


print(the_tree)

list_of_sentences = []
for i in tokenize.sent_tokenize(initial_sentence):
    list_of_sentences.append(tag.pos_tag(tokenize.word_tokenize(i)))
print(list_of_sentences)

x = Tree("ID:50", list_of_sentences)

print(x)
x.draw()

'''
for index, value in enumerate(the_tree):
    print("index" + str(index))
    print("value" + str(value))
    print(tokenize.word_tokenize(the_tree[index]))

    the_tree = Tree(tokenize.word_tokenize(the_tree[index]))

the_tree.draw()
'''
'''
print(tokenize.sent_tokenize(initial_sentence))
print("FOR LOOP")
for i in tokenize.sent_tokenize(initial_sentence):
    j = tokenize.word_tokenize(i)
    print(tag.pos_tag(j))
'''