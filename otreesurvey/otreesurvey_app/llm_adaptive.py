'''
VMP 10-12-2025:
This module contains functions and classes to facilitate an adaptive interview process.
'''

from openai import OpenAI
import os
from dotenv import load_dotenv
import instructor
from tenacity import retry, stop_after_attempt, wait_fixed
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

custom_retry = retry(
        stop=stop_after_attempt(5),      # Number of retries (change as needed)
        wait=wait_fixed(2),              # Wait 2 seconds between retries
        reraise=True                     # Raise the exception if all retries fail
        )

@custom_retry
def call_openai(response_model, content_prompt, model_name='gpt-4.1-2025-04-14', temp=0.7): # gpt-5-mini
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    client = instructor.from_openai(client, mode=instructor.Mode.TOOLS)

    kwargs = dict(
        model=model_name,
        messages=[{"role": "user", "content": content_prompt}],
        response_model=response_model,
    )

    if model_name not in ['o3', 'o3-mini', 'o4-mini']:
        kwargs['temperature'] = temp

    try:
        return client.chat.completions.create(**kwargs)
    except Exception as exc:
        print(f"Exception with model {model_name}: {exc}")
        raise

class UserAnswer(BaseModel):
    question: str
    answer: str
    
class InterviewTurn(BaseModel):
    interviewer_utterance: str = Field(..., description="The interviewer’s utterance, containing acknowledgment of the previous answer and a relevant follow-up question.")
    rationale: Optional[str] = Field(None, description="Short rationale explaining why this utterance is appropriate given the context.")

def generate_conversational_question(history: List[UserAnswer], n_rounds=8) -> InterviewTurn:
    conversation_str = ""
    for turn in history:
        conversation_str += f"Interviewer: {turn.question}\nParticipant: {turn.answer}\n\n"

    current_round = len(history) + 1

    system_prompt = f"""
Context: 
You are a thoughtful, empathetic, and curious interviewer exploring the meat-eating habits and motivations of an interviewee.

Current conversation: 
{conversation_str}

=*=*= 

Task Description: 

Interview objective: By the end of the conversation, the interviewer has to learn about the following: 
1) The participant's personal meat-eating habits.
2) The participant's personal motivations for eating meat and/or for reducing/avoiding it. 
3) The meat-eating habits of the interviewee's social contacts (e.g., family, friends, peers, community).
4) The motivations that the interviewee's social contacts have to eat or avoid meat. 

Follow this strategy to generate your next question:  
1) Assess which of the 4 coverage goals have been adequately addressed so far, and which ones still need more exploration. 
2.1) If the previous answer introduced something potentially (1) important, (2) interesting, or (3) unclear, formulate an elaboration question about this.
2.2) Otherwise, prefer asking about an uncovered goal from the list above. 
2.3) If one the the interview goals above is not important for the participant do not pursue it further.
3) For the last question, invite the participant to share anything important that has not yet come up.

Follow these guidelines when constructing your next question:
1) Acknowledge the participant’s last answer to show you are listening and value their input.
2) Respond naturally to what the participant has said: be curious, warm and non-judgmental. 
3) Ask one focused open question per turn (avoid asking about more than one thing and avoid leading phrasing).
4) Keep it concise: ~1 sentence acknowledging what they said, then 1 clear question.
5) Avoid moralizing, advice, assumptions, checklists, multiple-choice, or multi-part questions.

Safety note: In an extreme case where the interviewee *exlicitly* refuses to answer the question do not force the interviewee to answer. Instead move the interview forward by asking about another topic from the list above.

Conversation constraints:
- You have {n_rounds} total turns; this is round {current_round} of {n_rounds}.

Based on the current conversation generate the next interviewer question that best follows the strategy and guidelines above.
    """

    return call_openai(
        response_model=InterviewTurn,
        content_prompt=system_prompt,
        model_name='gpt-4.1-2025-04-14',
        temp=0.7
    )

def generate_pair_open_question(
    history: List[UserAnswer],
    pair: Tuple[str, str],
    pair_index: int,
    total_pairs: int,
    last_pair_answer: Optional[str] = None,
    last_connection_choice: Optional[str] = None,
) -> InterviewTurn:
    """
    Generate the next interviewer utterance that asks about how a specific pair of
    statements relate for the participant, given the previous interview history,
    and (optionally) the previous pair's answer + categorical choice.
    """
    # Turn the prior main interview into a readable transcript
    conversation_str = ""
    for turn in history:
        conversation_str += (
            f"Interviewer: {turn.question}\n"
            f"Participant: {turn.answer}\n\n"
        )

    label1, label2 = pair
    human_round = pair_index + 1  # 1-based for humans

    # Optional short block about the previous pair
    prev_block = ""
    if pair_index > 0 and (last_pair_answer or last_connection_choice):
        choice_text = {
            "support": "that they support each other",
            "conflict": "that they conflict with each other",
            "unclear": "that they are not clearly connected",
        }.get(last_connection_choice or "", last_connection_choice or "how they relate")

        prev_block = f"""
Most recent pair (pair {human_round - 1}) summary:
- Participant's open answer: "{last_pair_answer or '[no answer recorded]'}"
- Participant's categorical choice: {choice_text}.

You may acknowledge this briefly in one short sentence before moving on to the next pair.
"""

    system_prompt = f"""
You are a thoughtful, neutral, non-judgmental interviewer. Your task is to ask 
clear, open-ended follow-up questions about how pairs of things relate to each other.

MAIN INTERVIEW CONTEXT:
{conversation_str}

You are now in a follow-up interview focusing on how things relate for the participant.
You will ask about {total_pairs} pairs of things. This is pair {human_round} of {total_pairs}.

{prev_block}

The next pair is:
1. {label1}
2. {label2}

For THIS turn:

1) If this is pair 1 of {total_pairs}:
   - Do NOT acknowledge any previous pair.
   - Simply introduce the pair and ask one open question of the form:
     "Two things that came up in the interview were [x] and [y]. Are they 
      connected in some way for you? And if yes, how?"

2) If this is NOT pair 1:
   - The interviewee coded the last connection as either supporting, conflicting, or none of these.
   - Begin with ONE short, neutral acknowledgement of how they coded the previous pair and described it in their answer.
   - The acknowledgement may use natural language such as:
       * "That makes sense, thank you for describing that."
       * "Thank you for sharing how these things relate for you."
   - It may include interpretive language (e.g., "fit together", "tension", 
     "support", "conflict") **ONLY if the participant clearly implied or stated 
     that type of relationship** in their previous answer. Do not introduce 
     interpretations they did not give.
   - Acknowledgements must stay descriptive and grounded in their words, without 
     strengthening or reinterpreting their answer.

3) Then ask exactly ONE open question about how {label1} and {label2} relate for them.
   - Stay neutral: do NOT pre-assign the relationship (supporting, conflicting, etc.).
   - Do NOT introduce new interpretations or value judgments.
   - A safe pattern is:
     "Are these two things connected in some way for you? And if yes, how?"

4) Tone and style:
   - Warm but neutral and non-judgmental.
   - No moralizing, no praise, no advice.
   - Sound natural and conversational, not mechanical.
   - Keep it concise: 2–3 sentences total (acknowledgement + question).

Return only:
- interviewer_utterance: the full text you will say;
- rationale: a brief explanation of why it fits the guidelines.
"""

    return call_openai(
        response_model=InterviewTurn,
        content_prompt=system_prompt,
        model_name='gpt-4.1-2025-04-14',
        temp=0.7,
    )

def generate_pair_scale_question(
    history: List[UserAnswer],
    pair: Tuple[str, str],
    open_answer: str,
) -> InterviewTurn:
    conversation_str = ""
    for turn in history:
        conversation_str += (
            f"Interviewer: {turn.question}\n"
            f"Participant: {turn.answer}\n\n"
        )

    label1, label2 = pair

    system_prompt = f"""
You are a thoughtful, neutral, non-judgmental interviewer. Your task is to ask 
clear, open-ended follow-up questions about how pairs of things relate to each other.

MAIN INTERVIEW CONTEXT:
{conversation_str}

You are in a follow-up phase asking about how specific pairs of things relate.

Current pair:
1. {label1}
2. {label2}

The participant's open answer about how these two relate was:
"{open_answer}"

For THIS turn:

1) Start with ONE short, neutral acknowledgement of what they just said.
   - Acknowledgements may include mild interpretive language ("fit together", 
     "tension", "support", "conflict") but do do not introduce interpretations they did not give.
   - Do NOT intensify, simplify, or reinterpret their answer (“it makes sense that…”
     is acceptable as long as it does not add new meaning or justification).

2) Then ask them to briefly summarize how they would describe the connection 
   between these two things OVERALL.
   - This question will later be mapped to categories (support/conflict/unclear), 
     but you must NOT name these categories.
   - Ask a single neutral summarizing question such as:
     "If you had to choose, how would you describe the way these two things relate overall?"

3) The question must remain neutral:
   - Do NOT suggest a type of relationship.
   - Do NOT hint toward conflict, support, or ambiguity unless they explicitly described it.

4) Tone and style:
   - Warm but neutral and non-judgmental.
   - No moralizing, no praise, no advice.
   - Sound natural and conversational, not mechanical.
   - Keep it concise: 2–3 sentences total (acknowledgement + summary question).

Return only:
- interviewer_utterance: the full text you will say;
- rationale: a brief explanation of why it fits the guidelines.
"""

    return call_openai(
        response_model=InterviewTurn,
        content_prompt=system_prompt,
        model_name='gpt-4.1-2025-04-14',
        temp=0.7,
    )