from otree.api import *
import json, time, asyncio 
import random
from .llm_adaptive import *
from datetime import datetime
from otree.api import Page
import itertools # new import for network comparison
import math # new import for network comparison

# for local, on HEROKU needs to be set. 
from openai import AsyncOpenAI # OpenAI
from dotenv import load_dotenv
load_dotenv()
ASYNC_CLIENT = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# for the training
from dataclasses import dataclass
from typing import Dict, List, Tuple

doc = """
Your app description
"""

US_STATES = [
    'Not Applicable', 'Alaska', 'Alabama', 'Arkansas', 'Arizona',
    'California', 'Colorado', 'Connecticut', 'District of Columbia',
    'Delaware', 'Florida', 'Georgia', 'Hawaii',
    'Iowa', 'Idaho', 'Illinois', 'Indiana',
    'Kansas', 'Kentucky', 'Louisiana', 'Massachusetts',
    'Maryland', 'Maine', 'Michigan', 'Minnesota',
    'Missouri', 'Mississippi', 'Montana', 'North Carolina',
    'North Dakota', 'Nebraska', 'New Hampshire', 'New Jersey',
    'New Mexico', 'Nevada', 'New York', 'Ohio',
    'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island',
    'South Carolina', 'South Dakota', 'Tennessee', 'Texas',
    'Utah', 'Virginia', 'Vermont', 'Washington',
    'Wisconsin', 'West Virginia', 'Wyoming']

DISTRACTORS = [
    "My friends often go out to eat seahorse",
    "I eat beans to get enough cholesterol",
]

def stamp(player, label: str):
    """Append {'label': <string>, 'ts': <float>} to player's JSON log."""
    # 1) Load existing JSON (or empty list)
    try:
        arr = json.loads(player.page_timings_json or '[]')
        if not isinstance(arr, list):
            arr = []
    except Exception:
        arr = []
    # 2) Append a record; make sure 'label' is a string
    arr.append({'label': str(label), 'ts': time.time()})
    # 3) Save it back as JSON
    player.page_timings_json = json.dumps(arr)

# temporary for testing # 
from .interview_001 import RECORDED_QA
def preload_interview(player, qa_pairs):
    dummy_time = datetime.utcnow().isoformat()
    conversation = []

    # If qa_pairs is a dict, use .items(); if it's already a list of tuples, use it directly.
    iterator = qa_pairs.items() if isinstance(qa_pairs, dict) else qa_pairs

    for q, a in iterator:
        conversation.append({
            "question": q,
            "answer": a,
            "input_mode": "text",
            "time_sent": dummy_time,
            "time_received": dummy_time,
        })

    player.conversation_json = json.dumps(conversation)
    player.participant.vars["interview_turns"] = len(conversation)

# for trainining
@dataclass
class TrainingScenario:
    key: str
    name: str
    vignette_html: str
    train_stance_list: List[str]
    stance_id_to_text: Dict[int, str]
    allowed_relations: Dict[Tuple[int, int], List[str]]
    required_positive_pairs: List[Tuple[int, int]]
    required_negative_pairs: List[Tuple[int, int]]

class C(BaseConstants): 
    NAME_IN_URL = 'survey'
    PLAYERS_PER_GROUP = None 
    NUM_ROUNDS = 1 # is this a special variable or can I delete it?  
    MAX_TURNS = 8 # for the adaptive interview
    MAX_BELIEF_ITEMS = 30  # safely above max limimt.
    MEAT_FREQ_CATEGORIES = [
        "never",
        "less than once a week",
        "one or two days a week",
        "three or four days a week",
        "five or six days a week",
        "every day"
    ]
    NUM_NODES_THRESHOLD=3 
    NUM_NODES_MAX=10 

    # for pairwise questions
    TARGET_PER_BUCKET = 3  # we aim for up to 3 per polarity bucket
    N_PAIR_QUESTIONS = 3 * TARGET_PER_BUCKET  # upper bound, e.g. 9

    # --- Alex (your real example) ---
    TRAIN_TEXT_1 = (
        "Alex decided to learn Spanish to improve job prospects and to be able to "
        "speak with locals while traveling. To practice speaking, Alex occasionally "
        "attends a conversational meetup. Some of Alex’s colleagues are also learning "
        "Spanish and attending the meetup, and this motivates Alex to go as well. "
        "However, Alex feels very embarrassed speaking out loud, and this sometimes "
        "makes Alex avoid the meetup. To remain consistent, Alex tries to practice "
        "Spanish every day using a language-learning app. Alex has a busy job, and "
        "when it gets hectic Alex skips the daily practice."
    )

    STANCE_LIST_1 = [
        "Alex sees career benefits from learning Spanish",
        "Alex wants to speak with locals",
        "Alex practices daily with an app",
        "Alex has a busy job",
        "Alex attends a conversational meetup",
        "Colleagues are learning Spanish",
        "Alex feels embarrassed speaking out loud",
    ]

    STANCE_ID_TO_TEXT_1 = {
        1: STANCE_LIST_1[0],
        2: STANCE_LIST_1[1],
        3: STANCE_LIST_1[2],
        4: STANCE_LIST_1[3],
        5: STANCE_LIST_1[4],
        6: STANCE_LIST_1[5],
        7: STANCE_LIST_1[6],
    }

    RELATIONS_1 = {
        # 1 with others (career benefits)
        (1, 2): ["positive", "none"],                  # career benefits & locals: usually support
        (1, 3): ["positive", "none"],                  # app practice supports career benefits
        (1, 4): ["positive", "negative", "none"],      # busy job can increase importance or hinder progress, or be unrelated
        (1, 5): ["positive", "none"],                  # meetup helps learning → career benefits
        (1, 6): ["positive", "none"],                  # colleagues learning signals career value / motivation
        (1, 7): ["positive", "negative", "none"],      # embarrassment can motivate or hinder effort, or be unrelated

        # 2 with others (speak with locals)
        (2, 3): ["positive", "none"],                  # practice supports speaking with locals
        (2, 4): ["negative", "none"],                  # busy job undermines time/energy to speak
        (2, 5): ["positive"],                          # REQUIRED POS: meetup is direct speaking practice
        (2, 6): ["positive", "none"],                  # colleagues may motivate engagement
        (2, 7): ["negative", "none"],                  # embarrassment makes speaking harder

        # 3 with others (daily app practice)
        (3, 4): ["negative"],                          # REQUIRED NEG: busy job makes Alex skip practice
        (3, 5): ["positive", "none"],                  # meetup & app both help learning
        (3, 6): ["positive", "none"],                  # colleagues motivate practice
        (3, 7): ["positive", "negative", "none"],      # embarrassment can either push to practice more or avoid it

        # 4 with others (busy job)
        (4, 5): ["negative", "none"],                  # busy job makes it harder to attend meetup
        (4, 6): ["none"],                              # no systematic link in vignette
        (4, 7): ["none"],                              # no systematic link in vignette

        # 5 with others (meetup)
        (5, 6): ["positive"],                          # REQUIRED POS: colleagues at meetup motivate attendance
        (5, 7): ["negative"],                          # REQUIRED NEG: embarrassment leads to avoiding meetup

        # 6 with 7 (colleagues & embarrassment)
        (6, 7): ["positive", "negative", "none"],      # colleagues can reduce or increase embarrassment, or be neutral
    }

    REQUIRED_POS_1 = [(2, 5), (5, 6)]
    REQUIRED_NEG_1 = [(3, 4), (5, 7)]

    # --- Dummy 1 (Sam, simple) ---

    TRAIN_TEXT_2 = (
    "Jordan wants to improve their physical fitness and to have more energy during the day. "
    "A friend has invited Jordan to join their gym sessions, and the gym is located on "
    "Jordan’s way home from work, which makes it convenient to go. Jordan plans to go to "
    "the gym three times a week. However, Jordan often feels exhausted after work and "
    "feels self-conscious exercising in front of others, and these feelings sometimes "
    "lead Jordan to skip the gym."
    )
    
    STANCE_LIST_2 = [
    "Jordan wants to improve their physical fitness",          # 1
    "Jordan wants to have more energy during the day",         # 2
    "Jordan plans to go to the gym three times a week",        # 3
    "Jordan often feels exhausted after work",                 # 4
    "A friend has invited Jordan to join their gym sessions",  # 5
    "The gym is located on Jordan’s way home from work",       # 6
    "Jordan feels self-conscious exercising in front of others",  # 7
    ]

    STANCE_ID_TO_TEXT_2 = {
        1: STANCE_LIST_2[0],
        2: STANCE_LIST_2[1],
        3: STANCE_LIST_2[2],
        4: STANCE_LIST_2[3],
        5: STANCE_LIST_2[4],
        6: STANCE_LIST_2[5],
        7: STANCE_LIST_2[6],
    }


    REQUIRED_POS_2 = [
        (1, 3), # wanting better fitness + plan to go to gym
        (3, 5), # friend invitation + plan to go to gym
    ]

    REQUIRED_NEG_2 = [
        (3, 4), # gym plan + exhaustion after work
        (3, 7), # gym plan + self-consciousness 
    ]
    
    RELATIONS_2 = {
    # 1. Fitness goal with others
    (1, 2): ["positive", "none"],              # improving fitness and having more energy fit together
    (1, 3): ["positive"],                      # REQUIRED POS: plan clearly supports fitness
    (1, 4): ["negative", "none"],             # exhaustion makes fitness goal harder
    (1, 5): ["positive", "none"],             # friend invite helps with fitness goal
    (1, 6): ["positive", "none"],             # convenient gym helps with fitness goal
    (1, 7): ["negative", "none"],             # self-consciousness can undermine pursuing fitness

    # 2. Energy goal with others
    (2, 3): ["positive", "none"],             # plan to exercise supports energy goal
    (2, 4): ["negative", "none"],             # exhaustion conflicts with wanting more energy
    (2, 5): ["positive", "none"],             # friend invite supports exercising, so energy
    (2, 6): ["positive", "none"],             # convenient gym supports exercising, so energy
    (2, 7): ["negative", "none"],             # self-consciousness makes it harder to exercise, so less energy

    # 3. Gym plan with others
    (3, 4): ["negative"],                     # REQUIRED NEG: exhaustion makes it hard to follow the plan
    (3, 5): ["positive"],                     # REQUIRED POS: friend invite directly supports going
    (3, 6): ["positive", "none"],             # gym on the way home supports the plan
    (3, 7): ["negative"],                     # REQUIRED NEG: self-consciousness leads to skipping the gym

    # 4. Exhaustion with others
    (4, 5): ["positive", "negative", "none"], # friend might help (encouragement) or conflict (still too tired), or no clear link
    (4, 6): ["none"],                         # exhaustion & gym location: no necessary systematic relation here
    (4, 7): ["positive", "none"],             # feeling exhausted can increase self-consciousness about exercising

    # 5. Friend invitation with others
    (5, 6): ["positive", "none"],             # invitation + convenient location both make going easier
    (5, 7): ["positive", "negative", "none"], # friend may reduce self-consciousness (supportive) or increase it (comparison), or no clear effect

    # 6. Gym location with self-consciousness
    (6, 7): ["none"],                         # location itself doesn’t systematically change self-consciousness
    }

    # --- Dummy 2 (really simple) ---

    TRAIN_TEXT_3 = (
    "Riley wants to get more involved in the local community and registered for a weekly "
    "volunteer shift. A close friend also volunteers there, which makes Riley feel welcome. "
    "The community center is located close to Riley’s home, so getting there is usually easy. "
    "However, Riley sometimes needs to take care of a younger relative on short notice, which "
    "makes it difficult to attend the volunteer shift consistently. Riley tries to plan ahead "
    "each week, using reminders and a shared calendar, but unexpected caregiving needs sometimes "
    "interfere."
    )

    STANCE_LIST_3 = [
        "Riley wants to contribute to the local community",                # 1
        "Riley registered for a weekly volunteer shift",                   # 2
        "A close friend also volunteers there",                            # 3
        "The community center is close to Riley’s home",                   # 4
        "Riley sometimes needs to take care of a younger relative on short notice",  # 5
        "Riley tries to plan ahead using reminders and a shared calendar", # 6
        "Unexpected caregiving needs sometimes interfere with Riley’s plans",        # 7
    ]

    STANCE_ID_TO_TEXT_3 = {
        1: STANCE_LIST_3[0],
        2: STANCE_LIST_3[1],
        3: STANCE_LIST_3[2],
        4: STANCE_LIST_3[3],
        5: STANCE_LIST_3[4],
        6: STANCE_LIST_3[5],
        7: STANCE_LIST_3[6],
    }

    REQUIRED_POS_3 = [
        (2, 3),  # friend also volunteers + registered shift
        (2, 4),  # center is close + registered shift
    ]

    REQUIRED_NEG_3 = [
        (2, 5),  # caregiving on short notice + registered shift
        (2, 7),  # unexpected caregiving interference + registered shift
    ]

    RELATIONS_3 = {
        # 1. Community contribution goal
        (1, 2): ["positive", "none"],      # volunteering supports contributing
        (1, 3): ["positive", "none"],      # friend encouragement supports goal
        (1, 4): ["positive", "none"],      # easy location supports goal
        (1, 5): ["negative", "none"],      # caregiving makes contributing harder
        (1, 6): ["positive", "none"],      # planning supports contributing
        (1, 7): ["negative", "none"],      # unexpected conflicts undermine goal

        # 2. Registered volunteer shift (central action)
        (2, 3): ["positive"],              # REQUIRED POS: friend boosts attendance
        (2, 4): ["positive"],              # REQUIRED POS: location makes attending easy
        (2, 5): ["negative"],              # REQUIRED NEG: caregiving prevents attendance
        (2, 6): ["positive", "none"],      # planning helps maintain commitment
        (2, 7): ["negative"],              # REQUIRED NEG: disruptions conflict with shift

        # 3. Friend also volunteers
        (3, 4): ["positive", "none"],      # social + convenience both help attendance
        (3, 5): ["none"],                  # caregiving unrelated to friend
        (3, 6): ["positive", "none"],      # planning + social support both help
        (3, 7): ["none"],                  # disruptions unrelated to friend

        # 4. Location is close by
        (4, 5): ["none"],                  # caregiving unrelated to location
        (4, 6): ["positive", "none"],      # planning supports making use of location
        (4, 7): ["none"],                  # disruptions unrelated to distance

        # 5. Caregiving on short notice
        (5, 6): ["negative", "none"],      # planning may help but often cannot fix it
        (5, 7): ["positive"],              # caregiving increases unexpected disruptions

        # 6. Planning with reminders
        (6, 7): ["negative", "none"],      # planning reduces unexpected conflicts, though imperfect
    }

    TRAINING_SCENARIOS: dict[str, TrainingScenario] = {
        "example1": TrainingScenario(
            key="example1",
            name="Alex",
            vignette_html=TRAIN_TEXT_1,
            train_stance_list=STANCE_LIST_1,
            stance_id_to_text=STANCE_ID_TO_TEXT_1,
            allowed_relations=RELATIONS_1,
            required_positive_pairs=REQUIRED_POS_1,
            required_negative_pairs=REQUIRED_NEG_1,
        ),
        "example2": TrainingScenario(
            key="example2",
            name="Jordan",
            vignette_html=TRAIN_TEXT_2,
            train_stance_list=STANCE_LIST_2,
            stance_id_to_text=STANCE_ID_TO_TEXT_2,
            allowed_relations=RELATIONS_2,
            required_positive_pairs=REQUIRED_POS_2,
            required_negative_pairs=REQUIRED_NEG_2,
        ),
        "example3": TrainingScenario(
            key="example3",
            name="Riley",
            vignette_html=TRAIN_TEXT_3,
            train_stance_list=STANCE_LIST_3,
            stance_id_to_text=STANCE_ID_TO_TEXT_3,
            allowed_relations=RELATIONS_3,
            required_positive_pairs=REQUIRED_POS_3,
            required_negative_pairs=REQUIRED_NEG_3,
        ),
    }
    
    # VEMI questionnaire
    VEMI_ITEMS = [
        ("I want to be healthy", "H"),
        ("Plant-based diets are better for the environment", "E"),
        ("Animals do not have to suffer", "A"),
        ("Animals’ rights are respected", "A"),
        ("I want to live a long time", "H"),
        ("Plant-based diets are more sustainable", "E"),
        ("I care about my body", "H"),
        ("Eating meat is bad for the planet", "E"),
        ("Animal rights are important to me", "A"),
        ("Plant-based diets are environmentally-friendly", "E"),
        ("It does not seem right to exploit animals", "A"),
        ("Plants have less of an impact on the environment than animal products", "E"),
        ("I am concerned about animal rights", "A"),
        ("My health is important to me", "H"),
        ("I don’t want animals to suffer", "A"),
    ]
    
    # MEMI questionnaire
    MEMI_ITEMS = [
    ("It goes against nature to eat only plants.", "Natural"),
    ("Our bodies need the protein.", "Necessary"),
    ("I want to fit in.", "Normal"),
    ("It is delicious.", "Nice"),
    ("It makes people strong and vigorous.", "Necessary"),
    ("I don’t want other people to be uncomfortable.", "Normal"),
    ("It is in all of the best tasting food.", "Nice"),
    ("It could be unnatural not to eat meat.", "Natural"),
    ("It is necessary for good health.", "Necessary"),
    ("It is just one of the things people do.", "Normal"),
    ("It gives me pleasure.", "Nice"),
    ("I want to be sure I get all of the vitamins and minerals I need.", "Necessary"),
    ("Everybody does it.", "Normal"),
    ("It has good flavor.", "Nice"),
    ("It gives me strength and endurance.", "Necessary"),
    ("I don’t want to stand out.", "Normal"),
    ("Meals without it don’t taste good.", "Nice"),
    ("It is human nature to eat meat.", "Natural"),
    ("Eating meat is part of our biology.", "Natural"),
]

class Subsession(BaseSubsession):
    pass

class Group(BaseGroup):
    pass

class Player(BasePlayer):
    
    # Consent.html (used)
    consent_given = models.BooleanField(
        choices=[[True, 'I consent'], [False, 'I do not consent']],
        widget=widgets.RadioSelect,
        label=''
    ) 
    
    # MeatScale.html (used)
    meat_consumption_present = models.IntegerField(min=1, max=100)
    meat_consumption_past    = models.IntegerField(min=1, max=100)
    meat_consumption_future  = models.IntegerField(min=1, max=100)
    dissonance_personal = models.IntegerField(min=1, max=100)
    dissonance_social = models.IntegerField(min=1, max=100)
    
    # LLM nodes generation (used)
    prompt_used = models.LongStringField(blank=True) # "node_prompt"
    llm_result = models.LongStringField(blank=True) # "node_result"

    # LLM nodes (BeliefAccuracyRating.html)
    generated_nodes = models.LongStringField(blank=True) # used 
    generated_nodes_accuracy = models.LongStringField(blank=True) # used 
    generated_nodes_relevance = models.LongStringField(blank=True) # used 

    # process/filter nodes 
    final_nodes = models.LongStringField(blank=True) # used 
    num_nodes = models.IntegerField(initial=0) # used only during

    # for distractors (used)
    distractor_problem = models.BooleanField(initial=False)
    distractor_ratings = models.LongStringField(blank=True)

    # main task (used)
    positions_1 = models.LongStringField(blank=True) 
    positions_2 = models.LongStringField(blank=True)
    positions_3 = models.LongStringField(blank=True)

    edges_2 = models.LongStringField(blank=True)
    edges_3 = models.LongStringField(blank=True)
   
    # demographics (used)
    age = models.IntegerField(label='How old are you?', min=18, max=130) 
    gender = models.StringField(
        label='What is your gender?',
        choices=[
            "Female", 
            "Male", 
            "Non-binary", 
            "Prefer not to disclose", 
            "Other"],
        widget=widgets.RadioSelect
        ) 
    education = models.StringField(
        label='What is the highest level of school you have completed or the highest degree you have received?',
        choices=[
            "Less than high school degree", 
            "High school degree or equivalent (e.g., GED)",
            "Some college but no degree", 
            "Associate degree", 
            "Bachelor degree",
            "Graduate degree (e.g., Masters, PhD, M.D)"
            ],
        widget=widgets.RadioSelect
    ) 
    politics = models.StringField(
        label='How would you describe your political viewpoints?',
        choices=[
            "Very liberal",
            "Slightly liberal",
            "Moderate",
            "Slightly conservative",
            "Very conservative",
            "Prefer not to disclose"
            ],
        widget=widgets.RadioSelect
    ) 
    state = models.StringField(
        label="In which state do you currently live?",
        choices=US_STATES
    ) 
    zipcode = models.StringField(
        label="Please enter your 5-digit ZIP code:",
        min_length=5,
        max_length=5,
    ) 
    
    # This does not do anything now I believe.
    force_answer = models.BooleanField(initial=True)
    
    # Logging conversation (clean this up as well.) 
    conversation_json = models.LongStringField(initial="[]") # used 
    current_answer = models.LongStringField(blank=True) # not used in post-experiment
    voice_answer = models.LongStringField(blank=True) # not used in post-experiment
    interview_feedback = models.LongStringField(
        label="",
        blank=True
    ) # used in post-experiment
    interview_test = models.LongStringField(
        label="",
        blank=True
    ) # does not work 
    
    # page timings (used)
    page_timings_json = models.LongStringField(initial='[]')

    # test audio
    audio_data = models.LongStringField(blank=True) # allows blank
    
    # final feedback (used)
    final_feedback = models.LongStringField(label='', blank=True)
    
    # VEMI.html + MEMI.html
    vemi_responses = models.LongStringField(blank=True)
    memi_responses = models.LongStringField(blank=True)
    
    # edge prompts 
    llm_edge_prompt = models.LongStringField(blank=True) # used
    llm_edges = models.LongStringField(blank=True) # used 
    llm_edges_random = models.LongStringField(blank=True) # used 
    user_edges_random = models.LongStringField(blank=True) # used
    network_compare_results = models.LongStringField(blank=True)  # JSON log of 3 comparisons

    # form fields (per round)
    network_compare_choice = models.StringField(blank=True)   # not used in post
    network_compare_reason = models.LongStringField(blank=True) # not used in post 
    network_compare_mapping = models.StringField(blank=True)  # not used in post 
    network_compare_rating_1 = models.StringField(blank=True) # not used in post
    network_compare_rating_2 = models.StringField(blank=True) # not used in post 
    
    # Let's just try this: 
    conv_overall_0_100 = models.IntegerField(min=0, max=100)
    conv_overall_cat = models.IntegerField(
        choices=[
            (1, 'Terrible'),
            (2, 'Not good'),
            (3, 'Average / Neutral'),
            (4, 'Good'),
            (5, 'Excellent'),
        ]
    )

    # 2. Questions relevant
    conv_relevant_0_100 = models.IntegerField(min=0, max=100)
    conv_relevant_cat = models.IntegerField(
        choices=[
            (1, 'Strongly disagree'),
            (2, 'Somewhat disagree'),
            (3, 'Neither agree nor disagree'),
            (4, 'Somewhat agree'),
            (5, 'Strongly agree'),
        ]
    )

    # 3. Easy to express in chat
    conv_easy_chat_0_100 = models.IntegerField(min=0, max=100)
    conv_easy_chat_cat = models.IntegerField(
        choices=[
            (1, 'Strongly disagree'),
            (2, 'Somewhat disagree'),
            (3, 'Neither agree nor disagree'),
            (4, 'Somewhat agree'),
            (5, 'Strongly agree'),
        ]
    )

    # 4. Comfortable to be honest
    conv_comfort_0_100 = models.IntegerField(min=0, max=100)
    conv_comfort_cat = models.IntegerField(
        choices=[
            (1, 'Strongly disagree'),
            (2, 'Somewhat disagree'),
            (3, 'Neither agree nor disagree'),
            (4, 'Somewhat agree'),
            (5, 'Strongly agree'),
        ]
    )

    # 5. AI model felt creepy/intrusive
    conv_creepy_0_100 = models.IntegerField(min=0, max=100)
    conv_creepy_cat = models.IntegerField(
        choices=[
            (1, 'Strongly disagree'),
            (2, 'Somewhat disagree'),
            (3, 'Neither agree nor disagree'),
            (4, 'Somewhat agree'),
            (5, 'Strongly agree'),
        ]
    )

    # Optional open comment
    conv_open_feedback = models.LongStringField(blank=True)
    
    # Training main data
    training_order_json      = models.LongStringField(blank=True)
    training_nodes           = models.LongStringField(blank=True)
    training_positions_1     = models.LongStringField(blank=True)
    training_positions_2     = models.LongStringField(blank=True)
    training_positions_3     = models.LongStringField(blank=True)
    training_edges_2         = models.LongStringField(blank=True)  # positive only (last example)
    training_edges_3         = models.LongStringField(blank=True)  # pos+neg merged (last example)

    training_pos_retry_count = models.IntegerField(initial=0)
    training_neg_retry_count = models.IntegerField(initial=0)

    # Attempt logs (all examples)
    training_pos_attempts_json = models.LongStringField(initial='[]')
    training_neg_attempts_json = models.LongStringField(initial='[]')

    # Per-page logs (overwritten each time)
    training_pos_attempts_page = models.LongStringField(blank=True)
    training_neg_attempts_page = models.LongStringField(blank=True)

    # NEW: map page log (all examples + per-page buffer)
    training_map_attempts_json = models.LongStringField(initial='[]')
    training_map_attempts_page = models.LongStringField(blank=True)

    # Voice validation mode.
    plausibility_pairs_json = models.LongStringField(blank=True) # covered in json
    pair_open_responses_json = models.LongStringField(blank=True) # not used in post 
    pair_interview_json = models.LongStringField(blank=True, default="") # the main field
    pair_connection_choice = models.StringField(blank=True) # not used in post 

    # Testing that we are getting prolific IDs out
    prolific_pid = models.StringField(blank=True)
    prolific_study_id = models.StringField(blank=True)
    prolific_session_id = models.StringField(blank=True)

    # Testing getting exit pages out 
    exit_status = models.StringField(blank=True)
    last_page = models.StringField(blank=True)
    exit_url = models.StringField(blank=True)

for i in range(C.MAX_BELIEF_ITEMS):
    setattr(Player, f"belief_accuracy_{i}", models.IntegerField(blank=True))
    setattr(Player, f"belief_relevance_{i}", models.IntegerField(blank=True))

###### PAGES ######
class Progress_Interview_w1(Page):
    
    @staticmethod
    def vars_for_template(player: Player):
        return dict(percent=35)  

    @staticmethod
    def is_displayed(player):
        return player.num_nodes >= C.NUM_NODES_THRESHOLD and player.consent_given

    @staticmethod
    def before_next_page(player, timeout_happened):
        stamp(player, 'progress_interview:submit')

class Progress_Interview_w2(Page):
    
    @staticmethod
    def vars_for_template(player: Player):
        return dict(percent=25)  

    @staticmethod
    def is_displayed(player):
        return player.num_nodes >= C.NUM_NODES_THRESHOLD and player.consent_given

    @staticmethod
    def before_next_page(player, timeout_happened):
        stamp(player, 'progress_interview:submit')

# after the training.
class Progress_Practice_w1(Page):

    @staticmethod
    def vars_for_template(player: Player):
        return dict(percent=70)  

    @staticmethod
    def is_displayed(player):
        return player.num_nodes >= C.NUM_NODES_THRESHOLD and player.consent_given

    @staticmethod
    def before_next_page(player, timeout_happened):
        stamp(player, 'progress_practice:submit')

class Progress_Practice_w2(Page):

    @staticmethod
    def vars_for_template(player: Player):
        return dict(percent=45)  

    @staticmethod
    def is_displayed(player):
        return player.num_nodes >= C.NUM_NODES_THRESHOLD and player.consent_given

    @staticmethod
    def before_next_page(player, timeout_happened):
        stamp(player, 'progress_practice:submit')

# after doing their own networks: this does not take so long.
class Progress_Networks_w1(Page):

    @staticmethod
    def vars_for_template(player: Player):
        return dict(percent=90)  

    @staticmethod
    def is_displayed(player):
        return player.num_nodes >= C.NUM_NODES_THRESHOLD and player.consent_given

    @staticmethod
    def before_next_page(player, timeout_happened):
        stamp(player, 'progress_networks:submit')

class Progress_Networks_w2(Page):

    @staticmethod
    def vars_for_template(player: Player):
        return dict(percent=65)  

    @staticmethod
    def is_displayed(player):
        return player.num_nodes >= C.NUM_NODES_THRESHOLD and player.consent_given

    @staticmethod
    def before_next_page(player, timeout_happened):
        stamp(player, 'progress_networks:submit')

# after doing validation: this is ~25% with the interview.
class Progress_Validation_w2(Page):

    @staticmethod
    def vars_for_template(player: Player):
        return dict(percent=90)  

    @staticmethod
    def is_displayed(player):
        return player.num_nodes >= C.NUM_NODES_THRESHOLD and player.consent_given

    @staticmethod
    def before_next_page(player, timeout_happened):
        stamp(player, 'progress_validation:submit')

# CONSENT 
class Consent(Page):
    form_model = 'player'
    form_fields = ['consent_given']

    @staticmethod
    def vars_for_template(player: Player):
        stamp(player, 'consent:render')
        return dict(
            wave=player.session.config.get('wave', 'w1')  # default w1 if missing
        )

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        # logging things now
        participant = player.participant
        pid = participant.label or participant.vars.get('PROLIFIC_PID') or ''
        player.prolific_pid = pid

        # not sure how to log this right now
        player.prolific_study_id = (
            participant.vars.get('STUDY_ID')
            or participant.vars.get('study_id')
            or ''
        )
        player.prolific_session_id = (
            participant.vars.get('SESSION_ID')
            or participant.vars.get('session_id')
            or ''
        )

        stamp(player, 'consent:submit')
        player.force_answer = True

    def error_message(self, values):
        if values['consent_given'] is None:
            return "Please indicate whether you consent to participate."

# INTERVIEW 
class Information(Page):
    form_model = 'player'

    @staticmethod
    def is_displayed(player: Player): 
        return player.consent_given 
    
    @staticmethod 
    def before_next_page(player: Player, timeout_happened):
        stamp(player, 'information:submit')

# we should remove the audio data
# just douple check this.
class InterviewTest(Page):
    form_model = 'player'
    form_fields = ['interview_test', 'audio_data'] # testing audio

    @staticmethod
    def is_displayed(player: Player):
        return player.consent_given
    
    @staticmethod 
    def before_next_page(player: Player, timeout_happened):
        stamp(player, 'interviewtest:submit')

class InterviewMain(Page):
    form_model = 'player'
    form_fields = ['current_answer', 'voice_answer']

    @staticmethod
    def vars_for_template(player: Player):
        conversation = json.loads(player.conversation_json)

        if "interview_turns" not in player.participant.vars:
            player.participant.vars["interview_turns"] = 1

        if not conversation:
            initial_question = "In this interview, we're talking about meat-eating habits. Is this something you've thought much about, or not really?"
            conversation.append({
                "question": initial_question,
                "answer": "",
                "time_sent": datetime.utcnow().isoformat(),
                "time_received": None
            })
            player.conversation_json = json.dumps(conversation)

        return dict(
            conversation=conversation,
            current_turn=player.participant.vars["interview_turns"],
            max_turns=C.MAX_TURNS,
            progress_percentage=int(100 * player.participant.vars["interview_turns"] / C.MAX_TURNS)
        )

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        conversation = json.loads(player.conversation_json)

        # Determine response and input mode
        response = player.current_answer.strip() if player.current_answer else player.voice_answer.strip()
        input_mode = "text" if player.current_answer.strip() else "voice" if player.voice_answer.strip() else "unknown"

        # Always save the last given response if not empty
        if response:
            conversation[-1]["answer"] = response
            conversation[-1]["input_mode"] = input_mode
            conversation[-1]["time_received"] = datetime.utcnow().isoformat()
        else:
            # Fallback for debuging (should not happen with proper UI)
            conversation[-1]["answer"] = "[No response detected]"
            conversation[-1]["input_mode"] = "unknown"
            conversation[-1]["time_received"] = datetime.utcnow().isoformat()

        current_turn = player.participant.vars["interview_turns"]

        # Only append a new question if there are remaining turns
        if current_turn < C.MAX_TURNS:
            # Collect non-empty Q&A pairs only
            qa_history = [
                UserAnswer(question=entry["question"], answer=entry["answer"])
                for entry in conversation if entry.get("answer") and entry["answer"].strip()
            ]
            llm_turn = generate_conversational_question(qa_history, C.MAX_TURNS)

            conversation.append({
                "question": llm_turn.interviewer_utterance,
                "answer": "",  # start empty for next turn
                "time_sent": datetime.utcnow().isoformat()
            })

        # Save updated conversation
        player.conversation_json = json.dumps(conversation)
        player.participant.vars["interview_turns"] = current_turn + 1
        stamp(player, 'interviewmain:submit')

    @staticmethod
    def is_displayed(player: Player):
        return ( 
            player.participant.vars.get("interview_turns", 1) <= C.MAX_TURNS
            and player.consent_given
        )

class ConversationFeedback(Page):
    form_model = 'player'
    form_fields = [
        'conv_overall_0_100', 'conv_overall_cat',
        'conv_relevant_0_100', 'conv_relevant_cat',
        'conv_easy_chat_0_100', 'conv_easy_chat_cat',
        'conv_comfort_0_100', 'conv_comfort_cat',
        'conv_creepy_0_100', 'conv_creepy_cat',
        'conv_open_feedback',
    ]
    
    @staticmethod
    def is_displayed(player): 
        return player.consent_given
    
    # async live method 
    @staticmethod
    async def live_method(player, data):
        from .llm_utils import make_node_prompt

        async def call_and_parse(prompt, retries=3, delay=3):
            """Call OpenAI safely and ensure valid parsed list."""
            for attempt in range(retries):
                try:
                    completion = await ASYNC_CLIENT.chat.completions.create(
                        model="gpt-4.1-2025-04-14",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Respond ONLY with a JSON list of node objects under a key 'results'. "
                                    "No text outside JSON."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        stream=False,
                    )
                    raw = completion.choices[0].message.content.strip()

                    # Try JSON parsing
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        cleaned = raw.replace("```json", "").replace("```", "").strip()
                        parsed = json.loads(cleaned)

                    # Normalize structure
                    if isinstance(parsed, dict) and "results" in parsed:
                        return parsed["results"]
                    elif isinstance(parsed, list):
                        return parsed
                    else:
                        raise ValueError("Response not in expected JSON list format.")

                except Exception as e:
                    print(f"⚠️ Attempt {attempt+1} failed:", e)
                    if attempt < retries - 1:
                        await asyncio.sleep(delay)
                    else:
                        raise  # Final failure after retries

        try:
            # --- Build the prompt ---
            #  preload_interview(player, RECORDED_QA) # (YOU COMMENT OUT LATER)
            conversation = json.loads(player.conversation_json or "[]")
            qa = {
                e["question"]: e["answer"]
                for e in conversation
                if e.get("answer") and str(e["answer"]).strip()
            }

            prompt = make_node_prompt(qa)
            player.prompt_used = prompt

            # --- Run background call ---
            llm_nodes_list = await call_and_parse(prompt, retries=3, delay=2)

            # --- Store for later use (no save()) ---
            player.llm_result = json.dumps({"results": llm_nodes_list}, indent=2)
            player.generated_nodes = json.dumps(llm_nodes_list)
            player.num_nodes = len(llm_nodes_list)

            yield {player.id_in_group: {"done": True}}

        except Exception as e:
            print("❌ LLM permanently failed:", e)
            yield {player.id_in_group: {"done": False}}

    @staticmethod
    def before_next_page(player, timeout_happened):
        stamp(player, "conv_feedback:submit")


# ACCURACY RATINGS.
class BeliefAccuracyRating(Page):
    form_model = 'player'

    @staticmethod
    def _all_items(player: Player):
        """
        Real AI-generated nodes + fixed distractors, shuffled in a deterministic order
        per participant.
        """
        nodes = json.loads(player.generated_nodes or '[]')
        real = [{"belief": n.get("stance", ""), "is_distractor": False} for n in nodes]
        distractors = [{"belief": t, "is_distractor": True} for t in DISTRACTORS]
        items = real + distractors

        rnd = random.Random(player.participant.code)  # deterministic per participant
        rnd.shuffle(items)
        return items

    @staticmethod
    def get_form_fields(player: Player):
        items = BeliefAccuracyRating._all_items(player)
        # dynamic fields: belief_accuracy_0, belief_accuracy_1, ...
        return [f"belief_accuracy_{i}" for i in range(len(items))]

    @staticmethod
    def vars_for_template(player: Player):
        items = BeliefAccuracyRating._all_items(player)
        qa_pairs = json.loads(player.conversation_json or "[]")

        belief_items = []
        for i, it in enumerate(items):
            current_rating = player.field_maybe_none(f"belief_accuracy_{i}")
            belief_items.append(
                {
                    "index": i,
                    "belief": it["belief"],
                    "rating": current_rating,
                }
            )

        return dict(
            belief_items=belief_items,
            transcript=qa_pairs,
            C=C,
        )

    @staticmethod
    def error_message(player: Player, values):
        items = BeliefAccuracyRating._all_items(player)

        ratings_to_store = []
        missing = False

        for i, it in enumerate(items):
            belief = it["belief"]
            is_distractor = it["is_distractor"]
            rating = values.get(f"belief_accuracy_{i}", None)

            # store per-field value (0–100)
            setattr(
                player,
                f"belief_accuracy_{i}",
                int(rating) if rating not in (None, "") else None,
            )
            if rating in (None, ""):
                missing = True

            ratings_to_store.append(
                {
                    "belief": belief,
                    "rating": None if rating in (None, "") else int(rating),  # 0–100
                    "is_distractor": is_distractor,
                }
            )

        # Save all raw ACCURACY ratings for later use
        player.generated_nodes_accuracy = json.dumps(ratings_to_store)

        # Optional: keep distractor ratings separately
        player.distractor_ratings = json.dumps(
            [
                {"index": i, "belief": r["belief"], "rating": r["rating"]}
                for i, r in enumerate(ratings_to_store)
                if r["is_distractor"]
            ]
        )

        if missing:
            return "Please rate all items before continuing."

    @staticmethod
    def is_displayed(player: Player):
        return player.num_nodes >= C.NUM_NODES_THRESHOLD and player.consent_given

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        stamp(player, 'belief_accuracy:submit')

class BeliefRelevanceRating(Page):
    form_model = 'player'

    @staticmethod
    def get_form_fields(player: Player):
        items = BeliefAccuracyRating._all_items(player)
        # dynamic fields: belief_relevance_0, belief_relevance_1, ...
        return [f"belief_relevance_{i}" for i in range(len(items))]

    @staticmethod
    def vars_for_template(player: Player):
        items = BeliefAccuracyRating._all_items(player)
        qa_pairs = json.loads(player.conversation_json or "[]")

        belief_items = []
        for i, it in enumerate(items):
            current_rating = player.field_maybe_none(f"belief_relevance_{i}")
            belief_items.append(
                {
                    "index": i,
                    "belief": it["belief"],
                    "rating": current_rating,
                }
            )

        return dict(
            belief_items=belief_items,
            transcript=qa_pairs,
            C=C,
        )

    @staticmethod
    def error_message(player: Player, values):
        """
        1. Save relevance ratings (0–100) for all items.
        2. Then perform the FILTERING step, using ACCURACY ratings
           stored earlier in player.generated_nodes_accuracy.
        """
        items = BeliefAccuracyRating._all_items(player)

        missing = False
        relevance_to_store = []

        for i, it in enumerate(items):
            belief = it["belief"]
            is_distractor = it["is_distractor"]
            rating = values.get(f"belief_relevance_{i}", None)

            # store per-field relevance
            setattr(
                player,
                f"belief_relevance_{i}",
                int(rating) if rating not in (None, "") else None,
            )
            if rating in (None, ""):
                missing = True

            relevance_to_store.append(
                {
                    "belief": belief,
                    "relevance": None if rating in (None, "") else int(rating),
                    "is_distractor": is_distractor,
                }
            )

        # Save all raw RELEVANCE ratings (optional but useful)
        player.generated_nodes_relevance = json.dumps(relevance_to_store)

        if missing:
            return "Please rate how relevant each statement is before continuing."

        # ---------- FILTERING (based on ACCURACY ONLY) ----------
        accuracy_ratings = json.loads(player.generated_nodes_accuracy or "[]")

        # distractor_problem: any distractor with accuracy > 40
        distractor_problem = any(
            r["is_distractor"]
            and (r["rating"] is not None)
            and (r["rating"] > 40)
            for r in accuracy_ratings
        )
        player.distractor_problem = distractor_problem

        # keep only REAL items with ACCURACY ≥ 60
        kept = [
            r
            for r in accuracy_ratings
            if (not r["is_distractor"])
            and (r["rating"] is not None)
            and (r["rating"] >= 60)
        ]

        # take the top-N by accuracy
        MAX_KEEP = getattr(C, 'NUM_NODES_MAX', 10)
        kept.sort(key=lambda r: r["rating"], reverse=True)
        kept = kept[:MAX_KEEP]

        player.final_nodes = json.dumps(kept)
        player.num_nodes = len(kept)

    @staticmethod
    def is_displayed(player: Player):
        return player.num_nodes >= C.NUM_NODES_THRESHOLD and player.consent_given

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        stamp(player, 'belief_relevance:submit')
    

# TRAINING PAGES 
# [ insert page with more general description of training. ]
# [ but probably we will want to combine this ]
class TrainingBrief(Page):
    form_model = 'player'
    form_fields: list[str] = []

    @staticmethod
    def is_displayed(player: Player):
        return player.num_nodes >= C.NUM_NODES_THRESHOLD and player.consent_given

    @staticmethod
    def vars_for_template(player: Player):
        wave = player.session.config.get('wave', 'w1')  # default w1 if missing
        return dict(wave=wave)

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        stamp(player, "training_brief:submit")


# INTRO HELPERS 
def _get_training_order(player: Player) -> list[str]:
    pv = player.participant.vars

    # Only draw a fresh random order if not already present
    if 'training_order' not in pv:
        keys = list(C.TRAINING_SCENARIOS.keys())   # e.g. ["example1","example2","example3"]
        random.shuffle(keys)
        pv['training_order'] = keys

    return pv['training_order']

def _get_scenario_for_index(player: Player, idx: int) -> TrainingScenario:
    order = _get_training_order(player)  # will crash if not set
    key = order[idx]                     # IndexError if out of range -> good
    return C.TRAINING_SCENARIOS[key]     # KeyError if mis-specified -> good

def _training_intro_vars_for(player: Player, idx: int):
    scenario = _get_scenario_for_index(player, idx)
    return dict(
        training_items=scenario.train_stance_list,
        vignette_html=scenario.vignette_html,
        vignette_name=scenario.name,
        training_example_index=idx,
        training_example_key=scenario.key,
    )

def _training_intro_before_next_for(player: Player, idx: int, timeout_happened):
    scenario = _get_scenario_for_index(player, idx)
    nodes = [{"belief": b} for b in scenario.train_stance_list]
    player.training_nodes = json.dumps(nodes, ensure_ascii=False)

    # store full order once (only if not already stored)
    current = player.field_maybe_none('training_order_json')
    if not current:
        order = _get_training_order(player)
        player.training_order_json = json.dumps(order, ensure_ascii=False)

    stamp(player, f"training_intro_{idx}:submit")

# MAP HELPERS
def _training_map_vars_for(player: Player, idx: int):
    scenario = _get_scenario_for_index(player, idx)

    nodes = json.loads(player.training_nodes or '[]')
    labels = [n.get('belief') for n in nodes if n.get('belief')]

    return dict(
        belief_labels_json=json.dumps(labels, ensure_ascii=False),
        vignette_html=scenario.vignette_html,
        vignette_name=scenario.name,
        training_example_index=idx,
        training_example_key=scenario.key,
    )

def _training_map_before_next_for(player: Player, idx: int, timeout_happened):
    _merge_map_attempts(player, idx)
    scenario = _get_scenario_for_index(player, idx)
    stamp(player, f"training_map_{idx}:submit")

# POS HELPERS
def _training_pos_vars_for(player: Player, idx: int):
    scenario = _get_scenario_for_index(player, idx)

    positions = json.loads(player.training_positions_1)  # must exist
    pos_by_label = {p["label"]: p for p in positions}

    ordered_ids = sorted(scenario.stance_id_to_text.keys())
    labels = [scenario.stance_id_to_text[i] for i in ordered_ids]

    belief_points = [
        {"x": pos_by_label[label]["x"], "y": pos_by_label[label]["y"]}
        for label in labels
    ]

    allowed_map = {
        f"{a}-{b}": allowed
        for (a, b), allowed in scenario.allowed_relations.items()
    }
    required_pos = [f"{a}-{b}" for (a, b) in scenario.required_positive_pairs]

    return dict(
        belief_labels_json=json.dumps(labels, ensure_ascii=False),
        belief_points=json.dumps(belief_points),
        allowed_map_json=json.dumps(allowed_map),
        required_positive_json=json.dumps(required_pos),
        stance_id_to_text_json=json.dumps(scenario.stance_id_to_text),
        training_pos_retry_count=player.training_pos_retry_count,
        vignette_html=scenario.vignette_html,
        vignette_name=scenario.name,
        training_example_index=idx,
        training_example_key=scenario.key,
    )

def _training_pos_before_next_for(player: Player, idx: int, timeout_happened):
    _merge_pos_attempts(player, idx)
    scenario = _get_scenario_for_index(player, idx)
    stamp(player, f"training_edge_pos_{idx}:submit")

# NEG HELPERS
def _merge_map_attempts(player: Player, idx: int):
    raw_page = player.training_map_attempts_page or "[]"
    try:
        page_attempts = json.loads(raw_page)
    except:
        page_attempts = []

    raw_all = player.training_map_attempts_json or "[]"
    try:
        all_attempts = json.loads(raw_all)
    except:
        all_attempts = []

    for attempt in page_attempts:
        attempt["example_index"] = idx

    all_attempts.extend(page_attempts)

    player.training_map_attempts_json = json.dumps(all_attempts, ensure_ascii=False)
    player.training_map_attempts_page = ""
    
def _merge_pos_attempts(player: Player, idx: int) -> None:
    """
    Take the per-page JSON from `training_pos_attempts_page`, attach it to the
    global list in `training_pos_attempts_json`, and clear the per-page field.
    """
    raw_page = player.training_pos_attempts_page or "[]"
    try:
        page_attempts = json.loads(raw_page)
    except json.JSONDecodeError:
        page_attempts = []

    raw_all = player.training_pos_attempts_json or "[]"
    try:
        all_attempts = json.loads(raw_all)
    except json.JSONDecodeError:
        all_attempts = []

    # Optionally enforce example_index = idx (even though JS already sets it)
    for attempt in page_attempts:
        attempt["example_index"] = idx

    all_attempts.extend(page_attempts)

    player.training_pos_attempts_json = json.dumps(all_attempts, ensure_ascii=False)
    player.training_pos_attempts_page = ""


def _merge_neg_attempts(player: Player, idx: int) -> None:
    """
    Same idea for the negative page. Read from `training_neg_attempts_page`,
    append to `training_neg_attempts_json`, clear page field.
    """
    raw_page = player.training_neg_attempts_page or "[]"
    try:
        page_attempts = json.loads(raw_page)
    except json.JSONDecodeError:
        page_attempts = []

    raw_all = player.training_neg_attempts_json or "[]"
    try:
        all_attempts = json.loads(raw_all)
    except json.JSONDecodeError:
        all_attempts = []

    for attempt in page_attempts:
        attempt["example_index"] = idx

    all_attempts.extend(page_attempts)

    player.training_neg_attempts_json = json.dumps(all_attempts, ensure_ascii=False)
    player.training_neg_attempts_page = ""

def _training_neg_vars_for(player: Player, idx: int):
    scenario = _get_scenario_for_index(player, idx)

    # Positions after the positive page; these must exist at this point
    positions = json.loads(player.training_positions_2)
    pos_by_label = {p["label"]: p for p in positions}

    ordered_ids = sorted(scenario.stance_id_to_text.keys())
    labels = [scenario.stance_id_to_text[i] for i in ordered_ids]

    belief_points = [
        {"x": pos_by_label[label]["x"], "y": pos_by_label[label]["y"]}
        for label in labels
    ]

    prior_edges = json.loads(player.training_edges_2)

    allowed_map = {
        f"{a}-{b}": allowed
        for (a, b), allowed in scenario.allowed_relations.items()
    }
    required_neg = [f"{a}-{b}" for (a, b) in scenario.required_negative_pairs]

    return dict(
        belief_points=json.dumps(belief_points),
        belief_labels_json=json.dumps(labels, ensure_ascii=False),
        belief_edges_json=json.dumps(prior_edges),
        allowed_map_json=json.dumps(allowed_map),
        required_negative_json=json.dumps(required_neg),
        stance_id_to_text_json=json.dumps(scenario.stance_id_to_text),
        training_neg_retry_count=player.training_neg_retry_count,
        vignette_html=scenario.vignette_html,
        vignette_name=scenario.name,
        training_example_index=idx,
        training_example_key=scenario.key,
    )

def _training_neg_before_next_for(player: Player, idx: int, timeout_happened):
    _merge_neg_attempts(player, idx)
    scenario = _get_scenario_for_index(player, idx)
    stamp(player, f"training_edge_neg_{idx}:submit")

class TrainingIntro1(Page):
    template_name = 'otreesurvey_app/TrainingIntro.html'
    form_model = 'player'

    @staticmethod
    def is_displayed(player: Player):
        return player.num_nodes >= C.NUM_NODES_THRESHOLD and player.consent_given

    @staticmethod
    def vars_for_template(player: Player):
        return _training_intro_vars_for(player, 0)

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        _training_intro_before_next_for(player, 0, timeout_happened)

class TrainingIntro2(TrainingIntro1):
    template_name = 'otreesurvey_app/TrainingIntro.html'
    
    @staticmethod
    def vars_for_template(player: Player):
        return _training_intro_vars_for(player, 1)

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        _training_intro_before_next_for(player, 1, timeout_happened)

class TrainingIntro3(TrainingIntro1):
    template_name = 'otreesurvey_app/TrainingIntro.html'

    @staticmethod
    def vars_for_template(player: Player):
        return _training_intro_vars_for(player, 2)

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        _training_intro_before_next_for(player, 2, timeout_happened)

class TrainingMap1(Page):
    template_name = 'otreesurvey_app/TrainingMap.html'
    form_model = 'player'
    form_fields = ['training_positions_1', 'training_map_attempts_page']

    @staticmethod
    def is_displayed(player: Player):
        return player.num_nodes >= C.NUM_NODES_THRESHOLD and player.consent_given

    @staticmethod
    def vars_for_template(player: Player):
        return _training_map_vars_for(player, 0)

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        _training_map_before_next_for(player, 0, timeout_happened)


class TrainingMap2(TrainingMap1):
    @staticmethod
    def vars_for_template(player: Player):
        return _training_map_vars_for(player, 1)

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        _training_map_before_next_for(player, 1, timeout_happened)


class TrainingMap3(TrainingMap1):
    @staticmethod
    def vars_for_template(player: Player):
        return _training_map_vars_for(player, 2)

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        _training_map_before_next_for(player, 2, timeout_happened)

class TrainingPos1(Page):
    template_name = 'otreesurvey_app/TrainingPos.html'
    form_model = 'player'
    form_fields = [
        'training_positions_2',
        'training_edges_2',
        'training_pos_retry_count',
        'training_pos_attempts_page',
    ]

    @staticmethod
    def is_displayed(player: Player):
        return player.num_nodes >= C.NUM_NODES_THRESHOLD and player.consent_given

    @staticmethod
    def vars_for_template(player: Player):
        return _training_pos_vars_for(player, 0)

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        _training_pos_before_next_for(player, 0, timeout_happened)


class TrainingPos2(TrainingPos1):
    @staticmethod
    def vars_for_template(player: Player):
        return _training_pos_vars_for(player, 1)

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        _training_pos_before_next_for(player, 1, timeout_happened)


class TrainingPos3(TrainingPos1):
    @staticmethod
    def vars_for_template(player: Player):
        return _training_pos_vars_for(player, 2)

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        _training_pos_before_next_for(player, 2, timeout_happened)
    

class TrainingNeg1(Page):
    template_name = 'otreesurvey_app/TrainingNeg.html'
    form_model = 'player'
    form_fields = [
        'training_positions_3',
        'training_edges_3',
        'training_neg_retry_count',
        'training_neg_attempts_page',
    ]

    @staticmethod
    def is_displayed(player: Player):
        return player.num_nodes >= C.NUM_NODES_THRESHOLD and player.consent_given

    @staticmethod
    def vars_for_template(player: Player):
        return _training_neg_vars_for(player, 0)

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        _training_neg_before_next_for(player, 0, timeout_happened)


class TrainingNeg2(TrainingNeg1):
    @staticmethod
    def vars_for_template(player: Player):
        return _training_neg_vars_for(player, 1)

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        _training_neg_before_next_for(player, 1, timeout_happened)


class TrainingNeg3(TrainingNeg1):
    @staticmethod
    def vars_for_template(player: Player):
        return _training_neg_vars_for(player, 2)

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        _training_neg_before_next_for(player, 2, timeout_happened)


class MapNodePlacement(Page):
    form_model = 'player'
    form_fields = ['positions_1']

    @staticmethod
    def vars_for_template(player):
        # labels for the canvas
        labels = [
            item['belief']
            for item in json.loads(player.final_nodes or '[]')
            if item.get('belief')
        ]
        # Q/A transcript for the toggleable panel
        qa_pairs = json.loads(player.conversation_json or "[]")

        return dict(
            belief_labels_json=json.dumps(labels),
            transcript=qa_pairs,
        )

    @staticmethod
    def is_displayed(player: Player):
        return player.num_nodes >= C.NUM_NODES_THRESHOLD and player.consent_given

    # ✅ Async live method — EDGE extraction only, based on final_nodes
    @staticmethod
    async def live_method(player, data):
        from .llm_utils import make_edge_prompt

        # Only start when client says start/retry
        if not (data.get("start") or data.get("retry")):
            return

        async def call_and_parse(prompt, retries=3, delay=2):
            """Try the OpenAI call and ensure we return a list of edge dicts."""
            for attempt in range(retries):
                try:
                    completion = await ASYNC_CLIENT.chat.completions.create(
                        model="gpt-4.1-2025-04-14",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Respond ONLY with a JSON list of edge objects, "
                                    "no text outside JSON."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        stream=False,
                    )
                    raw = completion.choices[0].message.content.strip()

                    # Parse JSON safely
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        cleaned = raw.replace("```json", "").replace("```", "").strip()
                        parsed = json.loads(cleaned)

                    # Normalize
                    if isinstance(parsed, dict) and "results" in parsed:
                        return parsed["results"]
                    elif isinstance(parsed, list):
                        return parsed
                    else:
                        raise ValueError("Response not in expected JSON list format.")

                except Exception as e:
                    print(f"⚠️ Edge attempt {attempt+1} failed:", e)
                    if attempt < retries - 1:
                        await asyncio.sleep(delay)
                    else:
                        raise  # Final failure

        try:
            accepted_nodes = json.loads(player.final_nodes or "[]")

            conversation = json.loads(player.conversation_json or "[]")
            qa = {
                e["question"]: e["answer"]
                for e in conversation
                if e.get("answer") and str(e["answer"]).strip()
            }

            prompt = make_edge_prompt(qa, accepted_nodes)
            player.llm_edge_prompt = prompt

            llm_edges_list = await call_and_parse(prompt, retries=3, delay=2)
            player.llm_edges = json.dumps(llm_edges_list, indent=2)

            yield {player.id_in_group: {"done": True}}

        except Exception as e:
            print("❌ Edge LLM permanently failed:", e)
            yield {player.id_in_group: {"done": False}}

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        stamp(player, 'self_map:submit')


class MapEdgePos(Page):
    form_model = 'player'
    form_fields = ['positions_2', 'edges_2']

    @staticmethod
    def vars_for_template(player):
        labels = [item['belief'] for item in json.loads(player.final_nodes or '[]')]
        positions = json.loads(player.positions_1 or '[]')

        belief_points = [
            {"label": label, "x": positions[i]['x'], "y": positions[i]['y'], "radius": 15}
            for i, label in enumerate(labels)
        ]
    
        # Q/A transcript for the toggleable panel
        qa_pairs = json.loads(player.conversation_json or "[]")

        return dict(
            belief_points=belief_points,
            belief_labels_json=json.dumps(labels),
            belief_edges_json=json.dumps([]), 
            transcript=qa_pairs,
        )

    @staticmethod
    def is_displayed(player): 
        return (
            player.num_nodes >= C.NUM_NODES_THRESHOLD
            and player.consent_given
        )

    @staticmethod 
    def before_next_page(player, timeout_happened):
        stamp(player, 'self_edge_pos:submit')

class MapIntro(Page):
    @staticmethod
    def vars_for_template(player):
        # Final nodes with participants' own statements
        nodes = json.loads(player.final_nodes or "[]")
        own_statements = [
            n["belief"] for n in nodes
            if n.get("belief")
        ]

        # Interview transcript (same structure you use on other pages)
        qa_pairs = json.loads(player.conversation_json or "[]")

        return dict(
            transcript=qa_pairs,
            own_statements=own_statements,
        )

    @staticmethod
    def is_displayed(player):
        # Use whatever logic you want; this is an example
        return (
            player.num_nodes >= C.NUM_NODES_THRESHOLD
            and player.consent_given
        )

class MapEdgeNeg(Page):
    form_model = 'player'
    form_fields = ['positions_3', 'edges_3']

    @staticmethod
    def vars_for_template(player):
        labels = [item['belief'] for item in json.loads(player.final_nodes or '[]')]
        positions = json.loads(player.positions_2 or '[]')
        prior_edges = json.loads(player.edges_2 or '[]')  # positives from previous page

        belief_points = [
            {"label": label, "x": positions[i]['x'], "y": positions[i]['y'], "radius": 15}
            for i, label in enumerate(labels)
        ]

        # Q/A transcript for the toggleable panel
        qa_pairs = json.loads(player.conversation_json or "[]")
        
        return dict(
            belief_points=belief_points,
            belief_labels_json=json.dumps(labels),
            belief_edges_json=json.dumps(prior_edges),  
            transcript=qa_pairs,
        )

    @staticmethod
    def is_displayed(player): 
        return (
            player.num_nodes >= C.NUM_NODES_THRESHOLD
            and player.consent_given
        )

    @staticmethod 
    def before_next_page(player, timeout_happened):
        stamp(player, 'self_edge_neg:submit')

def _norm_pair(a, b):
    return (a, b) if a <= b else (b, a)


def compute_and_store_pairs(player):
    """
    Compute pairs of labels for plausibility + pair interview.

    Logic:
    1) If total possible label pairs <= N_PAIR_QUESTIONS, use ALL of them.
    2) Otherwise:
       a) Sample up to TARGET_PER_BUCKET from each bucket
          (positive, negative, none).
       b) If still < N_PAIR_QUESTIONS, top up randomly from remaining pairs.
    3) Shuffle and store; also log actual count in participant.vars.
    """

    # Reuse if already computed
    existing_json = player.field_maybe_none('plausibility_pairs_json')
    if existing_json:
        try:
            pairs = json.loads(existing_json)
            player.participant.vars.setdefault("n_pair_questions", len(pairs))
            return pairs
        except json.JSONDecodeError:
            # fall through to recompute if corrupted
            pass

    target_per_bucket = C.TARGET_PER_BUCKET
    max_pairs = C.N_PAIR_QUESTIONS

    nodes = json.loads(player.positions_3 or "[]")
    edges = json.loads(player.edges_3 or "[]")

    labels = [n.get("label") for n in nodes if n.get("label")]
    label_set = set(labels)

    # All unordered unique pairs of labels
    all_pairs = [(a, b) for i, a in enumerate(labels) for b in labels[i + 1:]]

    # === STEP 3 FIRST: if there are few possible pairs, just use all ===
    if len(all_pairs) <= max_pairs:
        random.shuffle(all_pairs)
        player.plausibility_pairs_json = json.dumps(all_pairs)
        player.participant.vars["n_pair_questions"] = len(all_pairs)
        return all_pairs

    # Otherwise, we have more than max_pairs: use bucket logic + top-up

    # Build edge lookup for polarity
    edge_lookup = {}
    for e in edges:
        a, b = e.get("stance_1"), e.get("stance_2")
        pol = e.get("polarity")
        if a and b and a in label_set and b in label_set:
            edge_lookup[_norm_pair(a, b)] = pol

    pos_pairs = [p for p in all_pairs if edge_lookup.get(_norm_pair(*p)) == "positive"]
    neg_pairs = [p for p in all_pairs if edge_lookup.get(_norm_pair(*p)) == "negative"]
    none_pairs = [p for p in all_pairs if _norm_pair(*p) not in edge_lookup]

    selected = []
    selected_set = set()

    def take(bucket):
        avail = [p for p in bucket if p not in selected_set]
        if avail:
            picks = random.sample(avail, min(target_per_bucket, len(avail)))
            selected.extend(picks)
            selected_set.update(picks)

    # 1) Try to fill buckets: up to TARGET_PER_BUCKET per bucket
    for bucket in [pos_pairs, neg_pairs, none_pairs]:
        take(bucket)

    # 2) Top up from remaining pairs to reach max_pairs if possible
    if len(selected) < max_pairs:
        remaining = [p for p in all_pairs if p not in selected_set]
        needed = max_pairs - len(selected)
        if remaining and needed > 0:
            extra = random.sample(remaining, min(needed, len(remaining)))
            selected.extend(extra)
            selected_set.update(extra)

    # 3) De-duplicate, shuffle, truncate to max_pairs
    selected = list(dict.fromkeys(selected))
    random.shuffle(selected)
    selected = selected[:max_pairs]

    player.plausibility_pairs_json = json.dumps(selected)
    player.participant.vars["n_pair_questions"] = len(selected)

    return selected

# generate the open question
class PairInterviewOpen(Page):
    form_model = 'player'
    form_fields = ['current_answer', 'voice_answer']

    @staticmethod
    def is_displayed(player: Player):
        # basic eligibility
        if not (player.num_nodes >= C.NUM_NODES_THRESHOLD and player.consent_given):
            return False

        # make sure pairs exist
        pairs = compute_and_store_pairs(player)
        total_pairs = len(pairs)
        if total_pairs == 0:
            return False

        # still have pairs left?
        current_idx = player.participant.vars.get("pair_index", 0)
        return current_idx < total_pairs

    @staticmethod
    def vars_for_template(player: Player):
        pairs = compute_and_store_pairs(player)
        total_pairs = len(pairs)

        pair_index = player.participant.vars.get("pair_index", 0)
        current_pair = pairs[pair_index]

        # load existing pair interview transcript
        raw = player.field_maybe_none("pair_interview_json") or "[]"
        try:
            pair_conv = json.loads(raw)
        except Exception:
            pair_conv = []

        # ---- info about previous pair for acknowledgement ----
        last_pair_answer = None
        last_connection_choice = None
        if pair_index > 0:
            prev_entries = [
                t for t in pair_conv if t.get("pair_index") == pair_index - 1
            ]
            if prev_entries:
                prev = prev_entries[-1]
                last_pair_answer = (prev.get("answer") or "").strip() or None
                last_connection_choice = (
                    prev.get("connection_choice") or ""
                ).strip() or None

        # ---- ensure we have an open question for THIS pair ----
        existing_for_pair = [
            t for t in pair_conv if t.get("pair_index") == pair_index
        ]
        has_open_question = any(
            (t.get("answer") or "").strip() == "" for t in existing_for_pair
        )

        if not existing_for_pair or not has_open_question:
            # build history from the main interview
            main_raw = player.field_maybe_none("conversation_json") or "[]"
            try:
                main_conv = json.loads(main_raw)
            except Exception:
                main_conv = []

            qa_history = [
                UserAnswer(question=e["question"], answer=e.get("answer", ""))
                for e in main_conv
                if e.get("answer") and str(e["answer"]).strip()
            ]

            llm_turn = generate_pair_open_question(
                history=qa_history,
                pair=current_pair,
                pair_index=pair_index,
                total_pairs=total_pairs,
                last_pair_answer=last_pair_answer,
                last_connection_choice=last_connection_choice,
            )

            new_entry = {
                "pair_index": pair_index,
                "pair": list(current_pair),
                "question": llm_turn.interviewer_utterance,
                "answer": "",
                "input_mode": None,
                "time_sent": datetime.utcnow().isoformat(),
                "time_received": None,
                "scale_question": None,
                "connection_choice": None,
            }
            pair_conv.append(new_entry)
            player.pair_interview_json = json.dumps(pair_conv)

        # ---- build conversation transcript for template ----
        label_map = {
            "support": "They support each other.",
            "conflict": "They conflict with each other.",
            "unclear": "They are not clearly connected.",
        }

        conversation_for_template = []
        for t in pair_conv:
            if t.get("pair_index", 0) <= pair_index:
                choice_code = (t.get("connection_choice") or "").strip()
                choice_text = label_map.get(choice_code, "")
                conversation_for_template.append(
                    {
                        "question": t["question"],
                        "answer": t.get("answer", ""),
                        "scale_question": t.get("scale_question", ""),
                        "connection_choice_text": choice_text,
                        "pair_index": t.get("pair_index"),
                        "pair": t.get("pair"),
                    }
                )

        current_turn = pair_index + 1
        max_turns = total_pairs
        progress_percentage = int(100 * current_turn / max_turns)

        return dict(
            conversation=conversation_for_template,
            current_turn=current_turn,
            max_turns=max_turns,
            progress_percentage=progress_percentage,
            current_pair_labels=current_pair,
        )

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        # load transcript
        raw = player.field_maybe_none("pair_interview_json") or "[]"
        try:
            pair_conv = json.loads(raw)
        except Exception:
            pair_conv = []

        pair_index = player.participant.vars.get("pair_index", 0)
        entries_for_pair = [t for t in pair_conv if t.get("pair_index") == pair_index]
        if not entries_for_pair:
            # shouldn't happen, but avoid crash
            stamp(player, "pair_open:submit:no_entry")
            return

        last_turn = entries_for_pair[-1]

        text = (player.current_answer or "").strip()
        voice = (player.voice_answer or "").strip()

        if text:
            response = text
            input_mode = "text"
        elif voice:
            response = voice
            input_mode = "voice"
        else:
            response = "[No response detected]"
            input_mode = "unknown"

        last_turn["answer"] = response
        last_turn["input_mode"] = input_mode
        last_turn["time_received"] = datetime.utcnow().isoformat()

        player.pair_interview_json = json.dumps(pair_conv)

        # IMPORTANT: do NOT increment pair_index yet
        # (scale page still uses this same pair)
        player.current_answer = ""
        player.voice_answer = ""

        stamp(player, "pair_open:submit")

# generate the scale question
class PairInterviewScale(Page):
    form_model = 'player'
    form_fields = ['pair_connection_choice']

    @staticmethod
    def is_displayed(player: Player):
        # basic eligibility
        if not (player.num_nodes >= C.NUM_NODES_THRESHOLD and player.consent_given):
            return False

        pairs = compute_and_store_pairs(player)
        total_pairs = len(pairs)
        if total_pairs == 0:
            return False

        pair_index = player.participant.vars.get("pair_index", 0)
        if pair_index >= total_pairs:
            return False

        # only show scale page if we already have an open answer
        raw = player.field_maybe_none("pair_interview_json") or "[]"
        try:
            pair_conv = json.loads(raw)
        except Exception:
            pair_conv = []

        entries_for_pair = [t for t in pair_conv if t.get("pair_index") == pair_index]
        if not entries_for_pair:
            return False

        last_turn = entries_for_pair[-1]
        ans = (last_turn.get("answer") or "").strip()
        return bool(ans)

    @staticmethod
    def vars_for_template(player: Player):
        pairs = compute_and_store_pairs(player)
        total_pairs = len(pairs)

        pair_index = player.participant.vars.get("pair_index", 0)
        current_pair = pairs[pair_index]

        # load existing convo
        raw = player.field_maybe_none("pair_interview_json") or "[]"
        try:
            pair_conv = json.loads(raw)
        except Exception:
            pair_conv = []

        # find entry for this pair (created on the open page)
        entries_for_pair = [t for t in pair_conv if t.get("pair_index") == pair_index]
        if not entries_for_pair:
            # safety; shouldn't happen
            scale_question = (
                "Thinking about these two things, how would you describe "
                "the way they relate overall?"
            )
        else:
            entry = entries_for_pair[-1]

            # if we already generated a scale question, reuse it
            scale_question = entry.get("scale_question")
            if not scale_question:
                # build history from main interview
                main_raw = player.field_maybe_none("conversation_json") or "[]"
                try:
                    main_conv = json.loads(main_raw)
                except Exception:
                    main_conv = []

                qa_history = [
                    UserAnswer(question=e["question"], answer=e.get("answer", ""))
                    for e in main_conv
                    if e.get("answer") and str(e["answer"]).strip()
                ]

                open_answer = entry.get("answer", "")

                llm_turn = generate_pair_scale_question(
                    history=qa_history,
                    pair=current_pair,
                    open_answer=open_answer,
                )

                scale_question = llm_turn.interviewer_utterance
                entry["scale_question"] = scale_question
                player.pair_interview_json = json.dumps(pair_conv)

        # ---- build conversation transcript (hide current scale question) ----
        label_map = {
            "support": "They support each other.",
            "conflict": "They conflict with each other.",
            "unclear": "They are not clearly connected.",
        }

        conversation_for_template = []
        for t in pair_conv:
            idx = t.get("pair_index", 0)
            choice_code = (t.get("connection_choice") or "").strip()
            choice_text = label_map.get(choice_code, "")

            # for previous pairs: show everything (including scale Q + choice)
            # for current pair: show ONLY open Q + open answer
            if idx < pair_index:
                scale_q = t.get("scale_question", "")
                conn_text = choice_text
            elif idx == pair_index:
                scale_q = ""        # avoid duplicate of current scale question
                conn_text = ""      # participant is choosing now
            else:
                # future pairs (shouldn't appear) – just in case
                scale_q = ""
                conn_text = ""

            conversation_for_template.append(
                {
                    "question": t["question"],
                    "answer": t.get("answer", ""),
                    "scale_question": scale_q,
                    "connection_choice_text": conn_text,
                    "pair_index": idx,
                    "pair": t.get("pair"),
                }
            )

        current_turn = pair_index + 1
        max_turns = total_pairs
        progress_percentage = int(100 * current_turn / max_turns)

        return dict(
            conversation=conversation_for_template,
            scale_question=scale_question,
            current_turn=current_turn,
            max_turns=max_turns,
            progress_percentage=progress_percentage,
            current_pair_labels=current_pair,
        )

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        raw = player.field_maybe_none("pair_interview_json") or "[]"
        try:
            pair_conv = json.loads(raw)
        except Exception:
            pair_conv = []

        pair_index = player.participant.vars.get("pair_index", 0)
        entries_for_pair = [t for t in pair_conv if t.get("pair_index") == pair_index]
        if not entries_for_pair:
            stamp(player, "pair_scale:submit:no_entry")
            return

        entry = entries_for_pair[-1]

        choice = (player.pair_connection_choice or "").strip()
        entry["connection_choice"] = choice
        entry["time_scale_answer"] = datetime.utcnow().isoformat()

        player.pair_interview_json = json.dumps(pair_conv)

        # now move on to the next pair
        player.participant.vars["pair_index"] = pair_index + 1
        player.pair_connection_choice = ""

        stamp(player, "pair_scale:submit")

class PairInterviewLLM(Page):
    form_model = 'player'
    form_fields = ['current_answer', 'voice_answer']

    @staticmethod
    def is_displayed(player: Player):
        if not (player.num_nodes >= C.NUM_NODES_THRESHOLD and player.consent_given):
            return False

        # Ensure pairs are computed (this will also set n_pair_questions)
        pairs = compute_and_store_pairs(player)
        total_pairs = len(pairs)

        if total_pairs == 0:
            return False

        current_idx = player.participant.vars.get("pair_index", 0)
        return current_idx < total_pairs

    @staticmethod
    def vars_for_template(player: Player):
        pairs = compute_and_store_pairs(player)
        total_pairs = len(pairs)

        # current pair index (0-based)
        pair_index = player.participant.vars.get("pair_index", 0)
        current_pair = pairs[pair_index]

        # Load existing pair interview transcript
        raw = player.field_maybe_none("pair_interview_json") or "[]"
        try:
            pair_conv = json.loads(raw)
        except Exception:
            pair_conv = []

        # Check if we already have a question for this pair with an empty answer
        existing_for_pair = [
            t for t in pair_conv if t.get("pair_index") == pair_index
        ]
        has_open_question = any(
            not (t.get("answer") or "").strip() for t in existing_for_pair
        )

        if not existing_for_pair or not has_open_question:
            # Build history from the main interview (player.conversation_json)
            main_raw = player.field_maybe_none("conversation_json") or "[]"
            try:
                main_conv = json.loads(main_raw)
            except Exception:
                main_conv = []

            qa_history = [
                UserAnswer(question=e["question"], answer=e.get("answer", ""))
                for e in main_conv
                if e.get("answer") and str(e["answer"]).strip()
            ]

            # Ask LLM for a question about this pair
            llm_turn = generate_pair_question(
                history=qa_history,
                pair=current_pair,
                pair_index=pair_index,
                total_pairs=total_pairs,
            )

            new_entry = {
                "pair_index": pair_index,
                "pair": list(current_pair),
                "question": llm_turn.interviewer_utterance,
                "answer": "",
                "input_mode": None,
                "time_sent": datetime.utcnow().isoformat(),
                "time_received": None,
            }
            pair_conv.append(new_entry)
            player.pair_interview_json = json.dumps(pair_conv)

        # Conversation to show: all pair turns up to & including current pair
        conversation_for_template = [
            {
                "question": t["question"],
                "answer": t.get("answer", ""),
                "pair_index": t.get("pair_index"),
                "pair": t.get("pair"),
            }
            for t in pair_conv
            if t.get("pair_index", 0) <= pair_index
        ]

        current_turn = pair_index + 1
        max_turns = total_pairs
        progress_percentage = int(100 * current_turn / max_turns)

        return dict(
            conversation=conversation_for_template,
            current_turn=current_turn,
            max_turns=max_turns,
            progress_percentage=progress_percentage,
            current_pair_labels=current_pair,
        )

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        # Load transcript
        raw = player.field_maybe_none("pair_interview_json") or "[]"
        try:
            pair_conv = json.loads(raw)
        except Exception:
            pair_conv = []

        pair_index = player.participant.vars.get("pair_index", 0)

        # Find the last question for this pair
        entries_for_pair = [t for t in pair_conv if t.get("pair_index") == pair_index]
        if not entries_for_pair:
            # should not happen – but avoid crashing
            stamp(player, "pair_interview_llm:submit:no_entry")
            player.participant.vars["pair_index"] = pair_index + 1
            return

        last_turn = entries_for_pair[-1]

        # Determine response and input mode
        text = (player.current_answer or "").strip()
        voice = (player.voice_answer or "").strip()

        if text:
            response = text
            input_mode = "text"
        elif voice:
            response = voice
            input_mode = "voice"
        else:
            response = "[No response detected]"
            input_mode = "unknown"

        last_turn["answer"] = response
        last_turn["input_mode"] = input_mode
        last_turn["time_received"] = datetime.utcnow().isoformat()

        # Save back
        player.pair_interview_json = json.dumps(pair_conv)

        # Increment pair index for next page
        player.participant.vars["pair_index"] = pair_index + 1

        # Clear form fields for next round
        player.current_answer = ""
        player.voice_answer = ""

        stamp(player, "pair_interview_llm:submit")

NETWORK_IDS = ['user', 'user_random', 'llm', 'llm_random']
ALL_PAIRS = list(itertools.combinations(NETWORK_IDS, 2))  # 6 pairs
FLIP_P = 0.5

def _flip_p(p):
    return {'positive': 'negative', 'negative': 'positive'}.get(p, p)

def choose_k(n, p):
    x = p * n
    lo = math.floor(x)
    hi = math.ceil(x)
    if lo == hi:
        return lo
    # choose hi with probability equal to fractional part
    return hi if random.random() < (x - lo) else lo

def flip_exact_fraction(raw_edges, p=0.5, *, pol_key='polarity'):
    """
    Flip exactly round(p * n) of the eligible signed edges (positive/negative),
    where n is the number of eligible edges.

    Returns a NEW list (does not mutate input list objects).
    """
    edges = [dict(e) for e in (raw_edges or [])]  # shallow copy each edge dict

    eligible = [i for i, e in enumerate(edges) if e.get(pol_key) in ('positive', 'negative')]
    n = len(eligible)
    if n == 0:
        return edges

    k = choose_k(n, p) # if uneven then randomly round up or down 
    k = max(0, min(n, k))

    flip_idx = set(random.sample(eligible, k))
    for i in flip_idx:
        edges[i][pol_key] = _flip_p(edges[i].get(pol_key))

    return edges

def ensure_user_random(player: Player):
    existing = player.field_maybe_none('user_edges_random')
    if existing:
        return

    raw = json.loads(player.edges_3 or "[]")
    flipped = flip_exact_fraction(raw, p=FLIP_P, pol_key='polarity')
    player.user_edges_random = json.dumps(flipped, indent=2, ensure_ascii=False)

def ensure_llm_random(player: Player):
    existing = player.field_maybe_none('llm_edges_random')
    if existing:
        return

    raw = json.loads(player.llm_edges or "[]")
    flipped = flip_exact_fraction(raw, p=FLIP_P, pol_key='polarity')
    player.llm_edges_random = json.dumps(flipped, indent=2, ensure_ascii=False)

def ensure_compare_plan(participant):
    """Random order of all pairs + random left/right, once per participant."""
    if participant.vars.get('compare_plan'):
        return

    pairs = ALL_PAIRS[:]   # now length 6
    random.shuffle(pairs)

    plan = []
    for a, b in pairs:
        left, right = (a, b) if random.random() < 0.5 else (b, a)
        plan.append({'left': left, 'right': right, 'pair': (a, b)})

    participant.vars['compare_plan'] = plan
    participant.vars['compare_index'] = 0
    participant.vars['compare_log'] = []

def _dedupe_signed_edges(raw_edges, label_set, *, from_key='stance_1', to_key='stance_2', pol_key='polarity'):
    """
    Keep at most one edge per unordered pair. If both positive & negative exist, prefer negative.
    Returns a list of dicts: {stance_1, stance_2, polarity}.
    """
    keep = {}
    for e in raw_edges or []:
        a = e.get(from_key); b = e.get(to_key); pol = e.get(pol_key)
        if not a or not b or a == b or pol not in ('positive','negative'):
            continue
        if a not in label_set or b not in label_set:
            continue
        key = tuple(sorted((a, b)))
        prev = keep.get(key)
        if not prev:
            keep[key] = {'stance_1': a, 'stance_2': b, 'polarity': pol}
        else:
            # If we already have one: prefer negative if either is negative
            if prev['polarity'] == 'negative' or pol == 'negative':
                keep[key] = {'stance_1': a, 'stance_2': b, 'polarity': 'negative'}
            else:
                # both positive -> keep as is
                pass
    return list(keep.values())

def edges_for(player: Player, net_id, label_set):
    if net_id == 'user':
        raw = json.loads(player.edges_3 or "[]")
        return _dedupe_signed_edges(raw, label_set)

    elif net_id == 'user_random':
        raw = json.loads(player.user_edges_random or "[]")
        return _dedupe_signed_edges(raw, label_set)

    elif net_id == 'llm':
        raw = json.loads(player.llm_edges or "[]")
        return _dedupe_signed_edges(raw, label_set)

    elif net_id == 'llm_random':
        raw = json.loads(player.llm_edges_random or "[]")
        return _dedupe_signed_edges(raw, label_set)

    return []

def _is_displayed_for(player: Player, idx: int):
    ok = bool(player.consent_given and player.positions_3 and player.llm_edges and player.edges_3)
    if not ok:
        return False

    ensure_llm_random(player)               # llm_random
    ensure_user_random(player)              # user_random
    ensure_compare_plan(player.participant) # 6-step plan

    return player.participant.vars.get('compare_index', 0) == idx

def _vars_for_template_for(player: Player, idx: int):
    # saved layout
    nodes = json.loads(player.positions_3 or "[]")
    labels = [n.get("label") for n in nodes if n.get("label")]
    label_set = set(labels)

    plan = player.participant.vars['compare_plan'][idx]
    left_id, right_id = plan['left'], plan['right']

    left_edges  = edges_for(player, left_id,  label_set)
    right_edges = edges_for(player, right_id, label_set)

    SRC_W, SRC_H = 585, 585  # original canvas size where positions_3 was recorded

    return dict(
        labels_json=json.dumps(labels),
        left_edges_json=json.dumps(left_edges),
        right_edges_json=json.dumps(right_edges),
        left_id_json=json.dumps(left_id),     # 'user' | 'llm' | 'llm_random'
        right_id_json=json.dumps(right_id),
        positions_src_json=json.dumps(nodes),
        positions_src_w=SRC_W,
        positions_src_h=SRC_H,
    )

def _before_next_page_for(player: Player, idx: int):
    plan = player.participant.vars['compare_plan'][idx]
    left_id, right_id = plan['left'], plan['right']

    choice = player.network_compare_choice  # '1' or '2'
    chosen_id = left_id if choice == '1' else (right_id if choice == '2' else None)

    row = dict(
        step_index  = idx,
        pair        = plan['pair'],
        left        = left_id,
        right       = right_id,
        choice      = choice,
        chosen_id   = chosen_id,
        reason      = (player.network_compare_reason or "").strip(),
        rating_left = player.network_compare_rating_1,
        rating_right= player.network_compare_rating_2,
    )

    log = player.participant.vars.get('compare_log', [])
    log.append(row)
    player.participant.vars['compare_log'] = log
    player.participant.vars['compare_index'] = idx + 1

    if idx == 5:  # last of 6
        player.network_compare_results = json.dumps(log, indent=2)

    stamp(player, f'network_compare_{idx}:submit')

class NetworkComparison1(Page):
    template_name = 'otreesurvey_app/NetworkComparison.html'
    form_model = 'player'
    form_fields = [
        'network_compare_choice',
        'network_compare_reason',
        'network_compare_mapping',
        'network_compare_rating_1',
        'network_compare_rating_2',
    ]
    @staticmethod
    def is_displayed(player: Player): return _is_displayed_for(player, 0)
    @staticmethod
    def vars_for_template(player: Player): return _vars_for_template_for(player, 0)
    @staticmethod
    def before_next_page(player: Player, timeout_happened): _before_next_page_for(player, 0)

class NetworkComparison2(NetworkComparison1):
    @staticmethod
    def is_displayed(player: Player): return _is_displayed_for(player, 1)
    @staticmethod
    def vars_for_template(player: Player): return _vars_for_template_for(player, 1)
    @staticmethod
    def before_next_page(player: Player, timeout_happened): _before_next_page_for(player, 1)

class NetworkComparison3(NetworkComparison1):
    @staticmethod
    def is_displayed(player: Player): return _is_displayed_for(player, 2)
    @staticmethod
    def vars_for_template(player: Player): return _vars_for_template_for(player, 2)
    @staticmethod
    def before_next_page(player: Player, timeout_happened): _before_next_page_for(player, 2)

class NetworkComparison4(NetworkComparison1):
    @staticmethod
    def is_displayed(player: Player): return _is_displayed_for(player, 3)
    @staticmethod
    def vars_for_template(player: Player): return _vars_for_template_for(player, 3)
    @staticmethod
    def before_next_page(player: Player, timeout_happened): _before_next_page_for(player, 3)

class NetworkComparison5(NetworkComparison1):
    @staticmethod
    def is_displayed(player: Player): return _is_displayed_for(player, 4)
    @staticmethod
    def vars_for_template(player: Player): return _vars_for_template_for(player, 4)
    @staticmethod
    def before_next_page(player: Player, timeout_happened): _before_next_page_for(player, 4)

class NetworkComparison6(NetworkComparison1):
    @staticmethod
    def is_displayed(player: Player): return _is_displayed_for(player, 5)
    @staticmethod
    def vars_for_template(player: Player): return _vars_for_template_for(player, 5)
    @staticmethod
    def before_next_page(player: Player, timeout_happened): _before_next_page_for(player, 5)

class MeatScale(Page):
    form_model = 'player'
    form_fields = [
        'meat_consumption_present',
        'meat_consumption_past',
        'meat_consumption_future',
        'dissonance_personal',
        'dissonance_social'
    ]

    @staticmethod
    def is_displayed(player: Player): 
        return (
            player.num_nodes >= C.NUM_NODES_THRESHOLD
            and player.consent_given
        )

    @staticmethod
    def error_message(player: Player, values):
        # Server-side safety: require all three to be provided
        if any(values.get(f) in (None, '') for f in [
            'meat_consumption_present',
            'meat_consumption_past',
            'meat_consumption_future',
        ]):
            return "Please move each slider to select a value."

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        stamp(player, 'meatscale:submit')    

class VEMI(Page):
    form_model = 'player'
    form_fields = ['vemi_responses']  # hidden JSON from the template

    @staticmethod
    def vars_for_template(player: Player):
        # Send items to the template (keys t/d are fine; JS handles them)
        items = [{"index": i + 1, "t": txt, "d": dom}
                 for i, (txt, dom) in enumerate(C.VEMI_ITEMS)]
        return dict(vemi_items_json=json.dumps(items))

    @staticmethod
    def error_message(player: Player, values):
        # Parse JSON, ensure every item has a 0–100 value (integer)
        raw = values.get('vemi_responses') or ''
        try:
            data = json.loads(raw)
        except Exception:
            return "There was a problem saving your answers. Please try again."

        if not isinstance(data, list) or len(data) != len(C.VEMI_ITEMS):
            return "Please answer every item before continuing."

        for row in data:
            v = row.get('value')
            if v is None:
                return "Please move every slider."
            try:
                iv = int(round(float(v)))
            except Exception:
                return "Please use the slider to select a value between 'Not Important' and 'Very Important'."
            if iv < 0 or iv > 100:
                return "Please use the slider to select a value between 'Not Important' and 'Very Important'."
            row['value'] = iv  # normalize to int 0–100

        # Store cleaned JSON
        player.vemi_responses = json.dumps(data)

    @staticmethod
    def is_displayed(player: Player): 
        return (
            player.num_nodes >= C.NUM_NODES_THRESHOLD
            and player.consent_given
        )
        
    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        stamp(player, 'questionnaire_vemi:submit')
    
class MEMI(Page):
    form_model = 'player'
    form_fields = ['memi_responses']  # just the hidden input

    @staticmethod
    def vars_for_template(player: Player):
        items = [{"index": i+1, "t": txt, "d": dom}
                for i, (txt, dom) in enumerate(C.MEMI_ITEMS)]
        return dict(memi_items_json=json.dumps(items))

    @staticmethod
    def error_message(player: Player, values):
        # Minimal validation: parse JSON, ensure every item has a 1..7 value.
        raw = values.get('memi_responses') or ''
        try:
            data = json.loads(raw)
        except Exception:
            return "There was a problem saving your answers. Please try again."

        if not isinstance(data, list) or len(data) != len(C.MEMI_ITEMS):
            return "Please answer every item before continuing."

        for row in data:
            v = row.get('value')
            if v is None:
                return "Please move every slider."
            try:
                iv = int(v)
            except Exception:
                return "Please use the slider to select a value between 'Not Important' and 'Very Important'."
            if iv < 0 or iv > 100:
                return "Please use the slider to select a value between 'Not Important' and 'Very Important'."
            row['value'] = iv  # normalize to int 0–100

        # Store the cleaned JSON
        player.memi_responses = json.dumps(data)
    
    @staticmethod
    def is_displayed(player: Player): 
        return (
            player.num_nodes >= C.NUM_NODES_THRESHOLD
            and player.consent_given
        )
        
    def before_next_page(player, timeout_happened):
        stamp(player, 'questionnaire_memi:submit')

class Demographics(Page): 
    form_model = 'player'
    form_fields = ['age', 'gender', 'education', 'politics', 'state', 'zipcode']

    @staticmethod
    def is_displayed(player: Player): 
        return (
            player.num_nodes >= C.NUM_NODES_THRESHOLD
            and player.consent_given
        )

    @staticmethod 
    def before_next_page(player: Player, timeout_happened):
        stamp(player, 'demographics:submit')

class Feedback(Page):
    form_model = 'player'
    form_fields = ['final_feedback']

    @staticmethod
    def is_displayed(player: Player): 
        return (
            player.num_nodes >= C.NUM_NODES_THRESHOLD
            and player.consent_given
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
        stamp(player, 'finalfeedback:submit')

class Results(Page):
    @staticmethod
    def is_displayed(player: Player): 
        return (
            player.num_nodes >= C.NUM_NODES_THRESHOLD
            and player.consent_given
        )

class Exit(Page): # gonna be outdated. 
    @staticmethod
    def is_displayed(player: Player):
        return (
            not player.consent_given
            or player.num_nodes < C.NUM_NODES_THRESHOLD
        )

### EXIT PAGES ### 
class LinkCompletion(Page):

    @staticmethod
    def is_displayed(player: Player):
        return (
            player.consent_given
            and player.num_nodes >= C.NUM_NODES_THRESHOLD
        )

    @staticmethod
    def vars_for_template(player: Player):
        player.exit_status = 'completed'
        player.last_page = 'LinkCompletion'
        player.exit_url = player.session.config['completionlink']
        stamp(player, 'exit:completed')
        return {}

    @staticmethod
    def js_vars(player: Player):
        return dict(
            url=player.session.config['completionlink']
        )

class LinkFailedChecks(Page):

    @staticmethod
    def is_displayed(player: Player):
        return (
            player.consent_given
            and player.num_nodes < C.NUM_NODES_THRESHOLD
        )

    @staticmethod
    def vars_for_template(player: Player):
        # Mark exit status as soon as the page is shown
        player.exit_status = 'failed_checks'
        player.last_page = 'LinkFailedChecks'
        player.exit_url = player.session.config['returnlink']
        stamp(player, 'exit:failed_checks')
        return {}

    @staticmethod
    def js_vars(player: Player):
        return dict(
            url=player.session.config['returnlink']
        )

class LinkNoConsent(Page):

    @staticmethod
    def is_displayed(player: Player):
        return not player.consent_given

    @staticmethod
    def vars_for_template(player: Player):
        player.exit_status = 'no_consent'
        player.last_page = 'LinkNoConsent'
        player.exit_url = player.session.config['noconsentlink']
        stamp(player, 'exit:no_consent')
        return {}

    @staticmethod
    def js_vars(player: Player):
        return dict(
            url=player.session.config['noconsentlink']
        )

# page sequence for wave 2: 
page_sequence = [
    Consent,
    LinkNoConsent, # if no consent return.
    Information, 
    InterviewTest, 
    *(InterviewMain for _ in range(C.MAX_TURNS)),
    ConversationFeedback,
    # a few post-interview things 
    BeliefAccuracyRating, 
    BeliefRelevanceRating,
    # ConversationFeedback,
    Progress_Interview_w2, 
    LinkFailedChecks, # used to be exit.
    ### NEXT PART ### 
    TrainingBrief,
    TrainingIntro1,
    TrainingMap1,
    TrainingPos1,
    TrainingNeg1,

    TrainingIntro2,
    TrainingMap2,
    TrainingPos2,
    TrainingNeg2,

    TrainingIntro3,
    TrainingMap3,
    TrainingPos3,
    TrainingNeg3,
    Progress_Practice_w2, 
    ### NEXT PART ###
    MapIntro,
    MapNodePlacement,
    MapEdgePos,  
    MapEdgeNeg, 
    Progress_Networks_w2, 
    *([PairInterviewOpen, PairInterviewScale] * C.N_PAIR_QUESTIONS),
    NetworkComparison1,
    NetworkComparison2,
    NetworkComparison3,
    Progress_Validation_w2, 
    MeatScale, 
    VEMI, 
    MEMI,  
    Demographics, 
    Feedback, 
    #Results,  
    LinkCompletion, # New thing to test. 
]

page_sequence = [
    Consent, 
    LinkNoConsent, 
    Information, 
    InterviewTest,
    *(InterviewMain for _ in range(C.MAX_TURNS)),
    ConversationFeedback,
    BeliefAccuracyRating, 
    BeliefRelevanceRating,
    LinkFailedChecks, 
    Progress_Interview_w2, 
    ### TRAINING ### 
    TrainingBrief,
    TrainingIntro1,
    TrainingMap1,
    TrainingPos1,
    TrainingNeg1,
    Progress_Practice_w2, 
    ### OWN MAPPING ###
    MapIntro,
    MapNodePlacement,
    MapEdgePos,  
    MapEdgeNeg, 
    Progress_Networks_w2,  
    ### VALIDATION ### 
    *([PairInterviewOpen, PairInterviewScale] * C.N_PAIR_QUESTIONS),
    NetworkComparison1, 
    NetworkComparison2, 
    NetworkComparison3, 
    NetworkComparison4, # new
    NetworkComparison5, # new
    NetworkComparison6, # new
    Progress_Validation_w2, 
    ### QUESTIONNAIRES ###
    MeatScale, # do we need this again?
    VEMI,
    MEMI,
    Demographics, # do we need this again?
    Feedback, 
    LinkCompletion, 
]

'''
# page sequence for wave 1: 
page_sequence = [
    Consent, 
    LinkNoConsent, 
    Information, 
    InterviewTest,
    *(InterviewMain for _ in range(C.MAX_TURNS)),
    ConversationFeedback,
    BeliefAccuracyRating, 
    BeliefRelevanceRating,
    LinkFailedChecks, 
    Progress_Interview_w1, 
    ### TRAINING ### 
    TrainingBrief,
    TrainingIntro1,
    TrainingMap1,
    TrainingPos1,
    TrainingNeg1,
    TrainingIntro2,
    TrainingMap2,
    TrainingPos2,
    TrainingNeg2,
    TrainingIntro3,
    TrainingMap3,
    TrainingPos3,
    TrainingNeg3,
    Progress_Practice_w1, 
    ### OWN MAPPING ###
    MapIntro,
    MapNodePlacement,
    MapEdgePos,  
    MapEdgeNeg, 
    Progress_Networks_w1,  
    ### QUESTIONNAIRES ###
    MeatScale, 
    Demographics, 
    Feedback, 
    LinkCompletion, 
]
'''
