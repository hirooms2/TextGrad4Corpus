import os
import random
import time
from openai import OpenAI
import textgrad as tg

OPENAI_API_KEY = " "
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

## 합친 prompt
problem_task_combined = """Problem Statement:
You will be given a few dialogs between users and a system, a recommended item relevant to their context.
Also, a list of passages is given.  
Your task is to refine the passages to better describe the features of the recommended item.  
The list of passages should comprehensively cover the item's features that users might show a positive preference for in any conversational context.  
Each passage should be concise, not overly long, and should begin with the title of the item, such as "Inception (2010), director Christopher Nolan."

Input:
A few dialogs and a recommended item.

[Dialogs]
{dialog}

[Recommended Items]
{target_item}

Output:
A list of passages"""

# passage corpus에서 추출한 passages
initial_solution = """
{passages}
"""

target_item = """Inception (2010)"""

dialogs = ["""System: Hello, how are you doing today?
User: Hello, I am doing well. How about you?
System: I'm doing great. Happy Thanksgiving! Are you interested in watching a movie trailer today?
User: Happy Thanksgiving to you too. Yes, I'm interested.
System: Great, what kind of movies do you like?
User: Action Suspense of the thriller kind.
System: Great. What was a recent movie you watched of that genre that you liked?
User: It was an older movie. It was called Frailty.
System: What did you like about that movie?
User: It kept you guessing as to whether the person he said he was was really that character. Or did he do a switch.""",
           """System: Hello, how are you doing today?
User: Hello, I am doing well. How about you?
System: I'm doing great. Happy Thanksgiving! Are you interested in watching a movie trailer today?
User: Happy Thanksgiving to you too. Yes, I'm interested.
System: Great, what kind of movies do you like?
User: Action Suspense of the thriller kind.
System: Great. What was a recent movie you watched of that genre that you liked?
User: It was an older movie. It was called Frailty.
System: What did you like about that movie?
User: It kept you guessing as to whether the person he said he was was really that character. Or did he do a switch.""",
           """System: Hi, I'm here to help recommend a movie for you.
User: Oh, alright. So, I am thinking about something that maybe is thought-provoking but also has fantasy elements.
System: What's one movie you liked that was similar to this?
User: Ok, so, I really like MCU and Pixar movies.
System: So you like more superhero and/or a decidedly good vs. evil kind of though provoking fantasy?
User: Doesn't have to have be good vs. evil, I also enjoy something with an interesting premise but doesn't throw it away in the middle of it.""",
           """System: Hi ho, you're looking for a movie?
User: Yes I am.
System: What's something you enjoyed recently?
User: I enjoy thriller or mystery movies.
System: I've enjoyed Christopher Nolan thrillers; what do you like about mysteries.
User: Oh nice he's the guy that made the batman movies right? I just like not knowing what is about to happen and then having to rewatch the movie so I can find out if I could have solved the mystery the first time watching it."""]

passage_list = """Passage 1. Inception (2010) actors Leonardo DiCaprio, Joseph Gordon-Levitt, Ellen Page, Tom Hardy – their performances intensify the suspense by deepening the mystery around their complex characters, keeping viewers emotionally engaged and guessing about their true identities.  

Passage 2. Inception (2010), director Christopher Nolan – Nolan’s masterful thriller storytelling keeps viewers guessing about character identities and plot twists, mirroring the user’s love for uncertainty and rewarding multiple viewings by revealing new layers of mystery.  

Passage 3. Inception (2010) genre Action, Adventure, Sci-Fi, Thriller – the film features a suspenseful, twisty plot with strong fantasy elements and a consistently thought-provoking premise that invites viewers to unravel its mysteries.  

Passage 4. Inception (2010) is celebrated for its intricate, multi-layered story structure that invites viewers to solve the mystery on repeat viewings, catching subtle clues and fully appreciating the film’s complex narrative and suspense elements."""

dialog_str = '\n\n'.join([f"Dialog {d_idx+1}\n{dialog}" for d_idx, dialog in enumerate(dialogs)])
problem_text_revision = problem_task_combined.format(dialog=dialog_str, target_item=target_item)
print(problem_text_revision)

initial_solution = initial_solution.format(passages=passage_list)
print(initial_solution)

llm_engine = tg.get_engine("gpt-4.1-mini")
tg.set_backward_engine(llm_engine)

# Passages is the variable of interest we want to optimize -- so requires_grad=True
passages = tg.Variable(value=initial_solution,
                       requires_grad=True,
                       role_description="passage list to optimize")

# We are not interested in optimizing the problem -- so requires_grad=False
problem = tg.Variable(problem_text_revision,
                      requires_grad=False,
                      role_description="passage revision problem")

# Let TGD know to update code!
optimizer = tg.TGD(parameters=[passages])

# The system prompt that will guide the behavior of the loss function.
loss_system_prompt = "You are a smart language model that evaluates a passage list. You do not change or add any passages, only evaluate the existing passage list critically and give very concise feedback."
loss_system_prompt = tg.Variable(loss_system_prompt, requires_grad=False, role_description="system prompt to the loss function")

# The instruction that will be the prefix
instruction_revision = """Instruction:
Think carefully about the dialog, the item, and the list of passages describing the features of the recommended item.
Identify the features that the user is likely to prefer based on the dialog, and evaluate whether each of these features is sufficiently described in the passages.
If a preferred feature is not covered by any passage, specify that feature.

Evaluation Steps:
1. Analyze the user's preferences from the dialog and express them as general features of the recommended item.
2. Determine whether each feature is sufficiently described in the passages.
3. If none of the passages cover a given feature, specify that feature.

Output Format:
Feature: [feature]
- Relevant passages: [passage numbers]
- Informativeness of these passages: [brief comment]"""

# Guidelines:
# - Focus on passages related to the user's preferences.
# - Do not evaluate unrelated passages."""


# The format string and setting up the call
format_string = "{instruction}\nProblem: {{problem}}\nCurrent Passage list: {{passages}}"
format_string = format_string.format(instruction=instruction_revision)
# print(format_string)

fields = {"problem": None, "passages": None}
formatted_llm_call = tg.autograd.FormattedLLMCall(engine=llm_engine,
                                                  format_string=format_string,
                                                  fields=fields,
                                                  system_prompt=loss_system_prompt)


# Finally, the loss function
def loss_fn(problem: tg.Variable, passages: tg.Variable) -> tg.Variable:
    inputs = {"problem": problem, "passages": passages}
    print("inputs: ", inputs, end="\n")

    return formatted_llm_call(inputs=inputs,
                              response_role_description=f"evaluation of the {passages.get_role_description()}")


# Let's do the forward pass for the loss function.
loss = loss_fn(problem, passages)
print(loss.value)  # 평가한 내용

# Let's look at the gradients!
loss.backward()
# breakpoint()
print(passages.gradients)  # 평가를 반영해서 어떻게 업데이트 할것인가? (gradient)

# Let's update the code
optimizer.step()  # optimize 대상 업데이트

print(passages.value)
