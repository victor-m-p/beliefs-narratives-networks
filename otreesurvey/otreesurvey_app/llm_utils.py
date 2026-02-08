from pydantic import BaseModel
from typing import List, Dict

# node prompt 
def make_node_prompt(questions_answers: Dict[str, str]) -> str:
    questions_answers_lines = "\n".join(
        f"- Q: {q}\n- A: {a}" for q, a in questions_answers.items()
    )

    prompt = f"""
Context: 
You are an expert analyst specializing in extracting key factors influencing meat consumption from interview transcripts.

Interview Transcript: 
{questions_answers_lines}

=*=*=

Task description: 
Analyze the interview transcript and extract **only** factors that influence the meat-eating habits of the interviewee. 
Extract both (1) factors that encourage or increase meat-eating, and (2) factors that discourage or reduce meat-eating.

For each factor, determine whether it relates to:
1) the personal behaviors and motivations of the interviewee, or
2) the behaviors and motivations of their social contacts (e.g., family, friends)

Only extract factors that relate to the behaviors and motivations of social contacts when these plausibly influence the meat-eating habits of the interviewee.

=*=*=

Extraction rules: 
1. Extract a maximum of 10 short statements (max 10 words each). If there are fewer than 10 relevant factors, extract only those that are clearly relevant.
2. Extract at least 1 statement about the personal meat-eating habits of the interviewee.
3. Provide the following fields for each statement: 
- **stance**: A concise summary of the attitude or behavior (no effects).
- **type**: Is the statement about the interviewee <PERSONAL> or about social contacts <SOCIAL>.
- **category**: Does the statement describe a <BEHAVIOR> or a <MOTIVATION>.
- **effect**: Does the statement describe something that <INCREASE> or a <DECREASE> the meat-eating habits of the interviewee.
4. If you cannot reliably assign an effect (increase or decrease), do not include the statement.

=*=*=

Extraction guidelines: 
1. Include only relevant influences:
    - Exclude anything that is unimportant, irrelevant, or does not clearly have an effect.
    - Example (EXCLUDE): "Health is not important to me."

2. Differentiate behavior from motivation:
    - Each statement should describe either a behavior or a motivation, not both.
    - **Behavior** = descriptions of behavior (e.g., "I never cook meat at home", "I eat beef every week").
    - **Motivation** = any reason to eat more or less meat (e.g., "I love the taste of meat", "I am concerned about animal welfare").
        - Example: "I avoid meat because of health concerns" should be split into two separate statements ("I avoid meat" and "I have health concerns").
        - Example: "Living with vegetarians limits meat cooking at home" should be split into two separate statements ("I live with vegetarians" and "I rarely cook meat at home").
        - Example: "I eat beef to get enough protein" should be split into two separate statements ("I eat beef" and "I want to get enough protein").
        - Example: "I was vegetarian but now eat meat for weight gain" should be split into two separate statements ("I was vegetarian" and "I eat meat" and "I want to gain weight").
        - Avoid [due to, because, as a result of, since, for, etc.] --- split into separate statements **very important**.

3. Ensure that each statement is well formed in isolation and can be evaluated by the interviewee as something they agree with or not: 
    - Each statement should be a clear, concise, and factual statement (max 10 words).
    - Each statement should be something that the interviewee plausibly agrees with. 
    - Think "I agree with the following: [statement]" where you take on the perspective of the interviewee.
    - Not good: "Financial limitations make it difficult" [unclear, cannot agree or disagree] --> good: "I have financial limitations"
    - Not good: "Availability of good vegetarian options" [unclear, cannot agree or disagree] --> good: "Good vegetarian options are limited"
    
=*=*=

Example:

Input transcript excerpt:
Q: Can you tell me what influences how much meat you eat?  
A: I love the taste of meat, but I live with vegetarians who are very concerned about animal welfare, so I rarely cook it. My family always serves meat when I visit. I also worry about the climate impact of eating too much meat.  

Expected output:
{{
"results": [
    {{
        "stance": "I love the taste of meat",
        "type": "PERSONAL",
        "category": "MOTIVATION",
        "effect": "INCREASE"
    }},
    {{
        "stance": "I live with vegetarians",
        "type": "SOCIAL",
        "category": "BEHAVIOR",
        "effect": "DECREASE"
    }},
    {{
        "stance": "Social contacts are concerned about animal welfare",
        "type": "SOCIAL",
        "category": "MOTIVATION",
        "effect": "DECREASE"
    }},
    {{
        "stance": "I rarely cook meat",
        "type": "PERSONAL",
        "category": "BEHAVIOR",
        "effect": "DECREASE"
    }},
    {{
        "stance": "My family serves meat during visits",
        "type": "SOCIAL",
        "category": "BEHAVIOR",
        "effect": "INCREASE"
    }},
    {{
        "stance": "I am concerned about climate change",
        "type": "PERSONAL",
        "category": "MOTIVATION",
        "effect": "DECREASE"
    }}
]
}}

=*=*=

### Output Format (JSON ONLY) ###
{{
"results": [
    {{
        "stance": "<concise summary of one motivation or behavior>",
        "type": "<PERSONAL or SOCIAL>",
        "category": "<BEHAVIOR or MOTIVATION>",
        "effect": "<INCREASE or DECREASE>"
    }}
]
}}
Return ONLY the JSON object, nothing else.
"""
    return prompt

class NodeModel(BaseModel): 
    stance: str
    type: str
    category: str
    effect: str

class NodeModelList(BaseModel): 
    results: List[NodeModel]

# edge prompt 
def make_edge_prompt(
    questions_answers: dict,
    nodes_list: list,
):
    # Format Q&A block
    questions_answers_lines = "\n".join(
        f"Q: {q}\nA: {a}" for q, a in questions_answers.items()
    )

    # Format the nodes
    belief_string = "\n".join(f"- {x}" for x in nodes_list)

    prompt = f"""
# Context:
You are an expert analyst specializing in understanding the relationships between beliefs, motivations, and behaviors related to meat consumption.
You are given a transcript of an interview along with a list of extracted stances (behaviors, beliefs, motivations, reasons, concerns, etc.).

# Interview transcript: 
{questions_answers_lines}

=*=*=

# Extracted stances:
{belief_string}

=*=*=

# Task description: 
Analyze the interview transcript and the list of extracted stances to identify meaningful connections between the stances.
You are looking for two types of relations: 
1. Supportive connections, where one stance reinforces or encourages another.
2. Conflicting connections, where one stance contradicts or undermines another. 
The goal is to build a network of how these stances relate to each other in terms of support or conflict.
The connections are undirected (i.e., A <-> B).

=*=*=

# Extraction rules:
1. Consider all unordered pairs of distinct stances (A, B). 
2. If there is **strong evidence** for a supportive or conflicting relationship that is felt by the interviewee include the undirected edge (A, B).
3. For **every** included edge:
   - Use **alphabetical order** for the two stances to avoid duplicates.
   - Set **polarity** to positive (supportive) or negative (conflicting).
   - Set **explicitness** to explicit or implicit depending on whether is **directly stated** or only **implied** in the transcript.
   - Assign a **strength** (1–100) based on the evidence in the transcript. Be conservative and use lower values if evidence is thin or ambiguous. Drop edges with weak evidence (<20).
   - Provide a concise **justification** for how (A, B) either support or conflict.

=*=*=

# Example:

Input transcript excerpt:
Q: Can you tell me what influences how much meat you eat?  
A: I love the taste of meat, but I live with vegetarians who are very concerned about animal welfare, so I rarely cook it. My family always serves meat when I visit. I also worry about the climate impact of eating too much meat.  

Extracted stances: 
- I love the taste of meat
- I live with vegetarians
- Social contacts are concerned about animal welfare
- I rarely cook meat
- My family serves meat during visits
- I am concerned about climate change

Expected output:

Expected output:
{{
"results": [
    {{
        "stance_1": "I live with vegetarians",
        "stance_1": "I love the taste of meat",
        "polarity": "negative",
        "explicitness": "explicit",
        "strength": 80,
        "justification": "Loving the taste of meat conflicts with living with vegetarians."
    }},
    {{
        "stance_1": "I live with vegetarians",
        "stance_1": "Social contacts are concerned about animal welfare",
        "polarity": "positive",
        "explicitness": "implicit",
        "strength": 70,
        "justification": "The vegetarian diet of the co-inhabitants of the interviewee is supported by their concern for animal welfare."
    }},
    {{
        "stance_1": "I rarely cook meat",
        "stance_1": "I live with vegetarians",
        "polarity": "positive",
        "explicitness": "explicit",
        "strength": 90,
        "justification": "It is explicitly stated that living with vegetarians supports the interviewee rarely cooking meat."
    }},
    {{
        "stance_1": "Social contacts are concerned about animal welfare", 
        "stance_1": "I rarely cook meat",
        "polarity": "positive",
        "explicitness": "explicit",
        "strength": 70,
        "justification": "It is explicitly stated that living with vegetarians who care about animal welfare supports the interviewee rarely cooking meat."
    }},
    {{
        "stance_1": "I love the taste of meat",
        "stance_1": "I rarely cook meat",
        "polarity": "negative",
        "explicitness": "explicit",
        "strength": 90,
        "justification": "Loving the taste of meat conflicts with rarely cooking meat."
    }},
    {{
        "stance_1": "I love the taste of meat",
        "stance_1": "My family serves meat during visits",
        "polarity": "positive",
        "explicitness": "implicit",
        "strength": 30,
        "justification": "Loving the taste of meat is plausibly implicitly supported by family environment"
    }},
    {{
        "stance_1": "I am concerned about climate change",
        "stance_1": "I rarely cook meat",
        "polarity": "positive",
        "explicitness": "explicit",
        "strength": 80,
        "justification": "Concern about climate change supports rarely cooking meat."
    }}
    {{
        "stance_1": "I love the taste of meat",
        "stance_1": "I am concerned about climate change",
        "polarity": "negative",
        "explicitness": "implicit",
        "strength": 50,
        "justification": "Loving the taste of meat conflicts with concern about climate change."
    }}
]
}}

=*=*=

# Output Format (JSON ONLY)
{{
  "results": [
    {{
      "stance_1": "<first node in alphabetical order>",
      "stance_2": "<second node in alphabetical order>",
      "polarity": "<positive|negative>",
      "explicitness": "<explicit|implicit>",
      "strength": <integer 1-100>,
      "justification": "<brief explanation>",
    }}
    // Repeat for each undirected edge
  ]
}}

ONLY return the JSON object, no additional text.
"""
    return prompt

class EdgeModel(BaseModel): 
        stance_1: str
        stance_2: str 
        polarity: str 
        explicitness: str 
        strength: int 
        justification: str
                
class EdgeModelList(BaseModel):
        results: List[EdgeModel]