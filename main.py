import os
import random
import time
from openai import OpenAI
import textgrad as tg

from prompts.refine_corpus_prompts import refine_prompt_single, refine_prompt_batch

OPENAI_API_KEY = ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

## 합친 prompt
problem_task_combined = refine_prompt_batch

# passage corpus에서 추출한 passages
initial_solution = """
{passages}
"""

target_item = """Inception (2010)"""


batch1 = ["""System: Hi ho, you're looking for a movie?
User: Yes I am.
System: What's something you enjoyed recently?
User: I enjoy thriller or mystery movies.
System: I've enjoyed Christopher Nolan thrillers; what do you like about mysteries.
User: Oh nice he's the guy that made the batman movies right? I just like not knowing what is about to happen and then having to rewatch the movie so I can find out if I could have solved the mystery the first time watching it.""",
          """System: Hi, I'm here to help recommend a movie for you.
User: Oh, alright. So, I am thinking about something that maybe is thought-provoking but also has fantasy elements.
System: What's one movie you liked that was similar to this?
User: Ok, so, I really like MCU and Pixar movies.
System: So you like more superhero and/or a decidedly good vs. evil kind of thought provoking fantasy?
User: Doesn't have to have be good vs. evil, I also enjoy something with an interesting premise but doesn't throw it away in the middle of it.""",
          """System: Hi, you're looking for a movie?
User: Hi, yes I am! I would love to watch a movie that's suspenseful and maybe a psychological thriller. Do you know of any movies like that?
System: Sure, what types of thrillers have you liked.
User: I really liked Shutter Island!
System: I did too. WHat did you like about it.
User: Well, I always love watching Leonardo DiCaprio because he's a great actor. But the plotline was interesting and had a lot of unexpected twists in it that I liked, especially the end. What did you like?
System: I like diCaprio too, and i agree that it was suspenseful. Do you like most of his movies.
User: I do like most of his movies. Do you have any recommendations for any of his other films you think I might like?"""]

batch2 = ["""System: Hello I hear you are looking for a movie trailer?
User: Yes I am.
System: what kind of movie are you wanting to see.
User: Some of my favorites are the Harry Potter Series, The Dark Night Trilogy, and Inception.""",
          """System: Hello!
User: Hey, how's it going? Ready to chat movies?
System: Yes! What kind of movies do you generally watch?
User: I tend to watch old and experimental stuff, like Memento.
System: Do you like to go to a movie theater or do you usually watch at home?
User: Normally at home.
System: What did you like about Memento?
User: It was a very interesting film, it had one big concept that everything was devoted to.
System: True, I enjoyed it as well. Is there any topics you like more than others? For example, do you like a love storyline along with the main storyline?
User: I tend to like Action films, but Romance and Foreign tend to work alright for me, genre wise.
System: Do you have a favorite action movie actor?
User: Tom Cruise.
System: Did you see him in Jack Reacher?
User: No, I wish I did! I liked him in Mission Impossible, like most other people.
System: That is a good one and one of his more recent movies. It was made in 2016, so not too old.
User: I don't really know of many other action movies I like. I guess Inception, that shares a director too.""",
          """System: Hello! I am here to help you look for a movie trailer.
User: ok!
System: What type of movies do you like to watch?
User: i like dramatic movies with good dialogue, and action. i like suspense in movies.
System: Would you say you like psychological thrillers?
User: yes, definately.
System: I do too! One of my favorites is Shutter Island, have you seen that one?
User: yes, i saw it awhile ago, i dont remember it too well. i like leonardo dicaprio.""",
          """System: Hello! Would you like to watch a movie trailer?
User: Yes.
System: What kind of movies do you like? Action/adventure, romance, comedy, drama?
User: I like drama and thriller. I love true crime and crime related or mystery action movies.
System: I think I know a movie you would enjoy. It's called Inception, with Leonardo DiCaprio. It's a mystery/thriller. Would you like to watch the trailer now?
User: Sure, but I've already seen that movie.
System: Let's try to find something else, then. How about a true crime story?
User: That sounds good."""]

dialogs = batch2

passage_list = """Passage 1. Inception (2010) lead actor Leonardo DiCaprio stars prominently, supported by Joseph Gordon-Levitt, Ellen Page, and Tom Hardy  
Passage 2. Inception (2010) director Christopher Nolan  
Passage 3. Inception (2010) genre Action, Adventure, Sci-Fi, Thriller, Mystery, Psychological Thriller  
Passage 4. Inception (2010) plot A suspenseful and mysterious story about a thief who uses dream-sharing technology to steal secrets and is tasked with planting an idea into a C.E.O.’s mind.  
Passage 5. Inception (2010) narrative style Features a layered story with mind-bending twists and a complex, non-linear structure that invites rewatching to uncover hidden clues.  
Passage 6. Inception (2010) psychological themes Explores dreams, subconscious mind manipulation, and the blurred lines between reality and illusion as central storytelling elements.  
Passage 7. Inception (2010) cinematic style Noted for striking visual effects that enhance its immersive experience.  
Passage 8. Inception (2010) soundtrack A memorable score that complements the film’s experimental and immersive atmosphere."""

dialog_str = '\n\n'.join([f"Dialog {d_idx+1}\n{dialog}" for d_idx, dialog in enumerate(dialogs)])
problem_text_revision = problem_task_combined.format(dialog=dialog_str, target_item=target_item, passages=passage_list)
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
