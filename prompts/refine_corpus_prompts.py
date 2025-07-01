
# 샘플 하나 처리용 프롬프트
refine_prompt_single="""Problem Statement:
You will be given a dialog between users and a system, a recommended item relevant to their context.
Also, a list of passages describing the item is also provided.  
Your task is to refine the passages to better describe the features of the recommended item.  
The final list of passages should comprehensively cover the item's features that users might show a positive preference for in various conversational contexts.
Each passage should focus on only one distinct feature of the item.
The passages must be concise, not overly long, and should begin with the title of the item, such as "Inception (2010) director Christopher Nolan."
Ensure that the passages are not redundant and that each describes a clearly separated and meaningful aspect of the item.

Input:
A dialog, a recommended item and a list of passages describing features of the item.

[Dialogs]
{dialog}

[Recommended Item]
{target_item}

[List of passages]
{passages}

Output:
A list of passages"""


# 미니 배치용 프롬프트
refine_prompt_batch = """Problem Statement:
You will be given multiple dialogs between users and a system, all associated with the same recommended item.
A list of passages describing the item is also provided.
Your task is to refine the passages to better describe the features of the recommended item in a way that is broadly relevant across diverse conversational contexts.
The final list of passages should comprehensively cover the item's features that users might positively respond to in any of the given dialogs.
Each passage should describe only one clearly distinct feature of the item, and redundant or overly specific content tied to a single dialog should be avoided.
The passage must be consice and must start with the item title such as "Inception (2010) director Christopher Nolan." 
Ensure that the passage list remains general yet expressive, avoiding repetition and covering a diverse but coherent range of item characteristics.


Input:
Multiple dialogs, a recommended item and a list of passages describing the item's features

[Dialogs]
{dialog}

[Recommended item]
{target_item}

[List of passages]
{passages}

Output:
A list of passages"""

