from sequence_tagging.model.data_utils import CoNLLDataset
from sequence_tagging.model.ner_model import NERModel
from sequence_tagging.config_seq import Config
import spacy
fr_nlp= spacy.load("fr")

def align_data(data):
    """Given dict with lists, creates aligned strings

    Adapted from Assignment 3 of CS224N

    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "

    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned



def interactive_shell(model):
    """Creates interactive shell to play with model

    Args:
        model: instance of NERModel

    """
    model.config.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
You can enter a review like
input> Accueil chaleureux mais prix cher""")

    while True:
        sentence = input("input> ")
        #sentence=remove_accent(sentence)
        sentence_nlp = fr_nlp(sentence)
        words_raw=[]
        words_raw.extend([sp.text for sp in sentence_nlp])

        #words_raw = sentence.strip().split(" ")
        #print(words_raw)

        if words_raw == ["exit"]:
            break

        preds,_ = model.predict(words_raw)
        to_print = align_data({"input": words_raw, "output": preds})

        for key, seq in to_print.items():
            model.config.logger.info(seq)


def main():
    # create instance of config
    config=Config()
    FLAGS=config.get_flags(".")
    # build model
    model = NERModel(config,FLAGS)
    model.build()
    model.restore_session(FLAGS.dir_model)

    # create dataset
    test  = CoNLLDataset(FLAGS.filename_test, config.processing_word,
                         config.processing_tag, FLAGS.max_iter)

    # evaluate and interact

    #results=model.evaluate_file(FLAGS.filename_samples,FLAGS.filename_aspects,fr_nlp)

    model.evaluate(test)
    model.print_results(test,FLAGS.filename_results)
    interactive_shell(model)


if __name__ == "__main__":
    main()