from summarizer import Summarizer
from keybert import KeyBERT
from flashtext import KeywordProcessor
from nltk.tokenize import sent_tokenize
import nltk
from nltk.corpus import wordnet as wn
from pywsd.similarity import max_similarity
from pywsd.lesk import adapted_lesk
from pywsd.lesk import simple_lesk
from pywsd.lesk import cosine_lesk
import re
import random
import streamlit as st

nltk.download('all')
nltk.download('wordnet')
model = Summarizer()
kw_model = KeyBERT(model='all-mpnet-base-v2')
# kw_model = KeyBERT()

full_text = """"The Greek historian knew what he was talking about. The Nile River fed Egyptian civilization for hundreds of years. The Longest River the Nile is 4,160 miles long—the world’s longest river. It begins near the equator in Africa and flows north to the Mediterranean Sea. In the south the Nile churns with cataracts. A cataract is a waterfall. Near the sea the Nile branches into a delta. A delta is an area near a river’s mouth where the water deposits fine soil called silt. In the delta, the Nile divides into many streams. The river is called the upper Nile in the south and the lower Nile in the north. For centuries, heavy rains in Ethiopia caused the Nile to flood every summer. The floods deposited rich soil along the Nile’s shores. This soil was fertile, which means it was good for growing crops. Unlike the Tigris and Euphrates, the Nile River flooded at the same time every year, so farmers could predict when to plant their crops. Red Land, Black Land The ancient Egyptians lived in narrow bands of land on each side of the Nile. They called this region the black land because of the fertile soil that the floods deposited. The red land was the barren desert beyond the fertile region. Weather in Egypt was almost always the same. Eight months of the year were sunny and hot. The four months of winter were sunny but cooler. Most of the region received only an inch of rain a year. The parts of Egypt not near the Nile were a desert. Isolation The harsh desert acted as a barrier to keep out enemies. The Mediterranean coast was swampy and lacked good harbors. For these reasons, early Egyptians stayed close to home. Each year, Egyptian farmers watched for white birds called ibises, which flew up from the south. When the birds arrived, the annual flood waters would soon follow. After the waters drained away, farmers could plant seeds in the fertile soil. Agricultural Techniques By about 2400 B.C., farmers used technology to expand their farmland. Working together, they dug irrigation canals that carried river water to dry areas. Then they used a tool called a shaduf to spread the water across the fields. These innovative, or new, techniques gave them more farmland. Egyptian Crops Ancient Egyptians grew a large variety of foods. They were the first to grind wheat into flour and to mix the flour with yeast and water to make dough rise into bread. They grew vegetables such as lettuce, radishes, asparagus, and cucumbers. Fruits included dates, figs, grapes, and watermelons. Egyptians also grew the materials for their clothes. They were the first to weave fibers from flax plants into a fabric called linen. Lightweight linen cloth was perfect for hot Egyptian days. Men wore linen wraps around their waists. Women wore loose, sleeveless dresses. Egyptians also wove marsh grasses into sandals. Egyptian Houses Egyptians built houses using bricks made of mud from the Nile mixed with chopped straw. They placed narrow windows high in the walls to reduce bright sunlight. Egyptians often painted walls white to reflect the blazing heat. They wove sticks and palm trees to make roofs. Inside, woven reed mats covered the dirt floor. Most Egyptians slept on mats covered with linen sheets. Wealthy citizens enjoyed bed frames and cushions. Egyptian nobles had fancier homes with tree-lined courtyards for shade. Some had a pool filled with lotus blossoms and fish. Poorer Egyptians simply went to the roof to cool off after sunset. They often cooked, ate, and even slept outside. Egypt’s economy depended on farming. However, the natural resources of the area allowed other economic activities to develop too. The Egyptians wanted valuable metals that were not found in the black land. For example, they wanted copper to make tools and weapons. Egyptians looked for copper as early as 6000 B.C. Later they learned that iron was stronger, and they sought it as well. Ancient Egyptians also desired gold for its bright beauty. The Egyptian word for gold was nub. Nubia was the Egyptian name for the area of the upper Nile that had the richest gold mines in Africa. Mining minerals was difficult. Veins (long streaks) of copper, iron, and bronze were hidden inside desert mountains in the hot Sinai Peninsula, east of Egypt. Even during the cool season, chipping minerals out of the rock was miserable work. Egyptians mined precious stones too. They were probably the first people in the world to mine turquoise. The Egyptians also mined lapis lazuli. These beautiful blue stones were used in jewelry.The Nile had fish and other wildlife that Egyptians wanted. To go on the river, Egyptians made lightweight rafts by binding together reeds. They used everything from nets to harpoons to catch fish. One ancient painting even shows a man ready to hit a catfish with a wooden hammer. More adventurous hunters speared hippopotamuses and crocodiles along the Nile. Egyptians also captured quail with nets. They used boomerangs to knock down flying ducks and geese. (A boomerang is a curved stick that returns to the person who threw it.) Eventually, Egyptians equipped their reed boats with sails and oars. The Nile then became a highway. The river’s current was slow, so boaters used paddles to go faster when they traveled north with the current. Going south, they raised a sail and let the winds that blew in that direction push them. The Nile provided so well for Egyptians that sometimes they had surpluses, or more goods than they needed. They began to trade with each other. Ancient Egypt had no money, so people exchanged goods that they grew or made. This method of trade is called bartering. Egypt prospered along the Nile. This prosperity made life easier and provided greater opportunities for many Egyptians. When farmers produce food surpluses, the society’s economy begins to expand. Cities emerge as centers of culture and power, and people learn to do jobs that do not involve agriculture. For example, some ancient Egyptians learned to be scribes, people whose job was to write and keep records. As Egyptian civilization grew more complex, people took on jobs other than that of a farmer or scribe. Some skilled artisans erected stone or brick houses and temples. Other artisans made pottery, incense, mats, furniture, linen clothing, sandals, or jewelry. A few Egyptians traveled to the upper Nile to trade with other Africans. These traders took Egyptian products such as scrolls, linen, gold, and jewelry. They brought back exotic woods, animal skins, and live beasts. As Egypt grew, so did its need to organize. Egyptians created a government that divided the empire into 42 provinces. Many officials worked to keep the provinces running smoothly. Egypt also created an army to defend itself. One of the highest jobs in Egypt was to be a priest. Priests followed formal rituals and took care of the temples. Before entering a temple, a priest bathed and put on special linen garments and white sandals. Priests cleaned the sacred statues in temples, changed their clothes, and even fed them meals. Together, the priests and the ruler held ceremonies to please the gods. Egyptians believed that if the gods were angry, the Nile would not flood. As a result, crops would not grow, and people would die. So the ruler and the priests tried hard to keep the gods happy. By doing so, they hoped to maintain the social and political order. Slaves were at the bottom of society. In Egypt, people became slaves if they owed a debt, committed a crime, or were captured in war. Egyptian slaves were usually freed after a period of time. One exception was the slaves who had to work in the mines. Many died from the exhausting labor. Egypt was one of the best places in the ancient world to be a woman. Unlike other ancient African cultures, in Egyptian society men and women had fairly equal rights. For example, they could both own and manage their own property. The main job of most women was to care for their children and home, but some did other jobs too. Some women wove cloth. Others worked with their husbands in fields or workshops. Some women, such as Queen Tiy, even rose to important positions in the government. Children in Egypt played with toys such as dolls, animal figures, board games, and marbles. Their parents made the toys from wood or clay. Boys and girls also played rough physical games with balls made of leather or reeds. Boys and some girls from wealthy families went to schools run by scribes or priests. Most other children learned their parents’ jobs. Almost all Egyptians married when they were in their early teens. As in many ancient societies, much of the knowledge of Egypt came about as priests studied the world to find ways to please the gods. Other advances came about because of practical discoveries. Egyptian priests studied the sky as part of their religion. About 5,000 years ago, they noticed that a star now called Sirius appeared shortly before the Nile began to flood. The star returned to the same position in 365 days. Based on that, Egyptians developed the world’s first practical calendar. The Egyptians developed some of the first geometry. Each year the Nile’s floods washed away land boundaries. To restore property lines, surveyors measured the land by using ropes that were knotted at regular intervals. Geometric shapes such as squares and triangles were sacred to Egyptians. Architects used them in the design of royal temples and monuments. Egyptian doctors often prepared dead bodies for burial, so they knew the parts of the body. That knowledge helped them perform some of the world’s first surgery. Some doctors specialized in using medicines made of herbs. Egyptian medicine was far from perfect. Doctors believed that the heart controlled thought and the brain circulated blood, which is the opposite of what is known now. Some Egyptian treatments would raise eyebrows today. One “cure” for an upset stomach was to eat a hog’s tooth crushed inside sugar cakes! Beginning about 3000 B.C., Egyptians developed a writing system using hieroglyphs. Hieroglyphs Hieroglyphs are pictures that stand for different words or sounds. Early Egyptians created a hieroglyphic system with about 700 characters. Over time the system grew to include more than 6,000 symbols. The Egyptians also developed a paperlike material called papyrus papyrus from a reed of the same name. Egyptians cut the stems into strips, pressed them, and dried them into sheets that could be rolled into scrolls. Papyrus scrolls were light and easy to carry. With them, Egyptians created some of the first books. Legend says a king named Narmer united Upper and Lower Egypt. Some historians think Narmer actually represents several kings who gradually joined the two lands. After Egypt was united, its ruler wore the Double Crown. It combined the red Crown of Lower Egypt with the white Crown of Upper Egypt. The first dynasty of the Egyptian empire began about 2925 B.C. A dynasty is a line of rulers from the same family. When a king died, one of his children usually took his place as ruler. The order in which members of a royal family inherit a throne is called the succession. More than 30 dynasties ruled ancient Egypt. Historians divide ancient Egyptian dynasties into the Old Kingdom, the Middle Kingdom, and the New Kingdom. The Old Kingdom started about 2575 B.C., when the Egyptian empire was gaining strength. The king of Egypt became known as the pharaoh pharaoh. The word pharaoh meant “great house,” and it was originally used to describe the king’s palace. Later it became the title of the king himself. The pharaoh ruled from the capital city of Memphis. The ancient Egyptians thought the pharaoh was a child of the gods and a god himself. Egyptians believed that if the pharaoh and his subjects honored the gods, their lives would be happy. If Egypt suffered hard times for a long period, the people blamed the pharaoh for angering the gods. In such a case, a rival might drive him from power and start a new dynasty. Because the pharaoh was thought to be a god, government and religion were not separate in ancient Egypt. Priests had much power in the government. Many high officials were priests. The first rulers of Egypt were often buried in an underground tomb topped by mud brick. Soon, kings wanted more permanent monuments. They replaced the mud brick with a small pyramid of brick or stone. A pyramid is a structure shaped like a triangle, with four sides that meet at a point. About 2630 B.C., King Djoser built a much larger pyramid over his tomb. It is called a step pyramid because its sides rise in a series of giant steps. It is the oldest-known large stone structure in the world. About 80 years later, a pharaoh named Khufu decided he wanted a monument that would show the world how great he was. He ordered the construction of the largest pyramid ever built. Along its base, each side was about 760 feet long. The core was built from 2.3 million blocks of stone. Building the Great Pyramid was hard work. Miners cut the huge blocks of stone using copper saws and chisels. These tools were much softer than the iron tools developed later. Other teams of workers pulled the stone slabs up long, sloping ramps to their place on the pyramid. Near the top of the pyramid, the ramps ended. Workers dragged each heavy block hundreds of feet and then set it in place. Farmers did the heavy labor of hauling stone during the season when the Nile flooded their fields. Skilled stonecutters and overseers worked year-round. The Great Pyramid took nearly 20 years to build. An estimated 20,000 Egyptians worked on it. A city called Giza was built for the pyramid workers and the people who fed, clothed, and housed them. Eventually, Egyptians stopped building pyramids. One reason is that the pyramids drew attention to the tombs inside them. Grave robbers broke into the tombs to steal the treasure buried with the pharaohs. Sometimes they also stole the mummies. Egyptians believed that if a tomb was robbed, the person buried there could not have a happy afterlife. During the New Kingdom, pharaohs began building more secret tombs in an area called the Valley of the Kings. The burial chambers were hidden in mountains near the Nile. This way, the pharaohs hoped to protect their bodies and treasures from robbers. Both the pyramids and later tombs had several passageways leading to different rooms. This was to confuse grave robbers about which passage to take. Sometimes relatives, such as the queen, were buried in the extra rooms. Tombs were supposed to be the palaces of pharaohs in the afterlife. Mourners filled the tomb with objects ranging from food to furniture that the mummified pharaoh would need. Some tombs contained small statues that were supposed to be servants for the dead person. Egyptian artists decorated royal tombs with wall paintings and sculptures carved into the walls. Art was meant to glorify both the gods and the dead person. A sculpture of a dead pharaoh had “perfect” features, no matter how he really looked. Artists also followed strict rules about how to portray humans. Paintings showed a person’s head, arms, and legs from the side. They showed the front of the body from the neck down to the waist. Wall paintings showed pharaohs enjoying themselves so they could have a happy afterlife. One favorite scene was of the pharaoh fishing in a papyrus marsh. Warlike kings were often portrayed in battle. Scenes might also show people providing for the needs of the dead person. Such activities included growing and preparing food, caring for animals, and building boats. As hard as the pharaohs tried to hide themselves, robbers stole the treasures from almost every tomb. Only a secret tomb built for a New Kingdom pharaoh was ever found with much of its treasure untouched. The dazzling riches found in this tomb show how much wealth the pharaohs spent preparing for the afterlife. By about 2130 B.C., Egyptian kings began to lose their power to local rulers of the provinces. For about 500 more years, the kings held Egypt together, but with a much weaker central government. This period of Egyptian history is called the Middle Kingdom. Rulers during the Middle Kingdom also faced challenges from outside Egypt. A nomadic people called the Hyksos invaded Egypt from the northeast. Their army conquered by using better weapons and horse-drawn chariots, which were new to Egyptians. After about 100 years, the Egyptians drove out the Hyksos and began the New Kingdom. """


# Distractors from Wordnet
def get_distractors_wordnet(syn,word):

    distractors = []
    word = word.lower()
    orig_word = word

    # replaces whitespace with underscore
    if len(word.split()) > 0:
        word = word.replace(" ", "_")

    hypernym = syn.hypernyms()
    if len(hypernym) == 0:
        return distractors

    for item in hypernym[0].hyponyms():
        name = item.lemmas()[0].name()
        #print ("name ",name, " word",orig_word)
        if name == orig_word:
            continue
        name = name.replace("_"," ")
        name = " ".join(w.capitalize() for w in name.split())
        if name is not None and name not in distractors:
            distractors.append(name)
    return distractors


def get_wordsense(sent, word):
    word = word.lower()

    if len(word.split()) > 0:
        word = word.replace(" ", "_")

    synsets = wn.synsets(word, 'n')
    if synsets:
        wup = max_similarity(sent, word, 'wup', pos='n')
        adapted_lesk_output = adapted_lesk(sent, word, pos='n')
        lowest_index = min(synsets.index(wup), synsets.index(adapted_lesk_output))
        return synsets[lowest_index]
    else:
        return None

def test(full_text):

    #region Text Summarizer
    result = model(full_text, min_length=60, max_length=500, ratio=0.4)
    summarized_text = ''.join(result)
    #endregion

    #region Keyword Extraction
    keywords = kw_model.extract_keywords(full_text, keyphrase_ngram_range=(1, 1), stop_words=None, highlight=False,
                                         top_n=10)
    keywords_list = list(dict(keywords).keys())
    important_keyword = []
    for keyword in keywords_list:
        if keyword.lower() in summarized_text.lower():
            important_keyword.append(keyword)
    #endregion

    #region Sentence Mapping

    sentences = sent_tokenize(summarized_text)
    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]

    keyword_processor = KeywordProcessor()
    keyword_sentences = {}
    for word in important_keyword:
        keyword_sentences[word] = []
        keyword_processor.add_keyword(word)

    for sentence in sentences:
        keywords_found = keyword_processor.extract_keywords(sentence)
        for key in keywords_found:
            keyword_sentences[key].append(sentence)


    for key in keyword_sentences.keys():
        values = keyword_sentences[key]
        values = sorted(values, key=len, reverse=True)
        keyword_sentences[key] = values

    #endregion

    #region Generate MCQ and Choices
    key_distractor_list = {}

    for keyword in keyword_sentences:
        wordsense = get_wordsense(keyword_sentences[keyword][0], keyword)
        if wordsense:
            distractors = get_distractors_wordnet(wordsense, keyword)
            #if len(distractors) == 0:
                #distractors = get_distractors_conceptnet(keyword)
            if len(distractors) != 0:
                key_distractor_list[keyword] = distractors
        else:

            #distractors = get_distractors_conceptnet(keyword)
            if len(distractors) != 0:
                key_distractor_list[keyword] = distractors


    #endregion

    #region printing
    index = 1
    for each in key_distractor_list:
        sentence = keyword_sentences[each][0]
        pattern = re.compile(each, re.IGNORECASE)
        output = pattern.sub(" _______ ", sentence)
        st.write("%s)" % (index), output)
        #print("%s)" % (index), output)
        choices = [each.capitalize()] + key_distractor_list[each]
        top4choices = choices[:4]
        #print("Correct answer is", top4choices[0])
        st.write("Correct answer is", top4choices[0])
        random.shuffle(top4choices)
        optionchoices = ['a', 'b', 'c', 'd']
        #print("\t", optionchoices[idx], ")", " ", choice)
        for idx, choice in enumerate(top4choices):
            st.write("\t", optionchoices[idx], ")", " ", choice)
        index = index + 1
    #endregion printing
    # return key_distractor_list
    # return keyword_sentences
    # return important_keyword
    # return summarized_text



