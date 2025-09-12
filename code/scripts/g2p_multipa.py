import os
from datasets import load_dataset
from whipa_utils import parse_langs, transliterate

def gen_from_iterable_dataset(iterable_ds):
    # suggested by https://stackoverflow.com/questions/76227219/can-i-convert-an-iterabledataset-to-dataset
    yield from iterable_ds

if __name__=="__main__":
    CORPUS = "multipa"
    for lg in parse_langs('all', CORPUS)[CORPUS]:
        print("Preprocessing", lg)
        if lg in ["hu", "pl", "ta"]: print("This process will take a while...")
        g2p = dict()
        for dsplit in ["train", "validation", "test"]:
            print(dsplit)
            cv_data = load_dataset("mozilla-foundation/common_voice_11_0",
                                    lg, split=dsplit, streaming=True)
            cv_data = cv_data.remove_columns(["accent", "age", "audio", "client_id",
                            "path", "down_votes", "gender", "segment", "up_votes"])
            if lg == "ta":
                cv_data = cv_data.filter(lambda sent: "ச" not in sent, input_columns="sentence")
            # transliterate
            cv_data = cv_data.map(transliterate).remove_columns(['locale'])
            # get sentence: ipa mapping
            g2p[dsplit] = {sample['sentence']: sample['ipa'] for sample in cv_data}
        # save each lg to file?
        if not os.path.isdir("../../data/multipa/ipa/"):
            os.mkdir("../../data/multipa/ipa/")

        with open("../../data/multipa/ipa/cv_{}_ipa.txt".format(lg), "w") as f:
            f.write(str(g2p))