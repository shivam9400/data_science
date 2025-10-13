import numpy as np

import numpy as np

def load_dataset():
    # Expanded dataset with over 200 examples
    sentences = [
        # --- Group 1: Product Reviews (Positive) --- (Approx. 25 examples)
        "I love this product, it works great.",
        "Absolutely fantastic service.",
        "This is the best I've used, highly recommend.",
        "Very satisfied and happy with my purchase.",
        "Highly recommend this to everyone.",
        "The performance exceeded my expectations by a mile.",
        "A truly wonderful purchase experience.",
        "Couldn't be happier with the results; flawless.",
        "This software is incredibly user-friendly and intuitive.",
        "I would definitely buy this again without hesitation.",
        "It arrived quickly and works perfectly right out of the box.",
        "Such a smooth and pleasant transaction.",
        "The quality is top-notch, far exceeding the price.",
        "Five stars! Everything about this is amazing.",
        "My new favorite gadget, totally indispensable.",
        "I am thoroughly impressed with the durability.",
        "It completely solved the problem I was having.",
        "A brilliant invention; simple yet effective.",
        "The design is sleek and modern.",
        "Exactly what I was looking for, spot on!",
        "It makes my work so much easier now.",
        "Outstanding value for the money.",
        "Delighted with the vibrant colors.",
        "The customer service was excellent and helpful.",
        "A perfect 10/10 product.",

        # --- Group 2: Product Reviews (Negative) --- (Approx. 25 examples)
        "I hate this, it broke immediately after one use.",
        "Terrible experience, would not recommend at all.",
        "Awful, I am severely disappointed and frustrated.",
        "Not good, very poor quality and construction.",
        "Worst product I have ever purchased in my life.",
        "The instructions were confusing, vague, and unhelpful.",
        "It stopped working after only a single day.",
        "I am extremely unsatisfied with the outcome.",
        "The customer support was rude, slow, and uncooperative.",
        "Do not waste your money on this terrible item.",
        "A massive letdown, I definitely regret buying this.",
        "This is totally unreliable and defective.",
        "Misleading advertisement; the picture was not accurate.",
        "It was missing parts upon arrival.",
        "I had to return it immediately because it was faulty.",
        "The product scratched easily and looks terrible now.",
        "My expectations were completely unmet.",
        "This felt cheap and flimsy.",
        "I'm deeply regretting this impulse buy.",
        "A complete failure of design and function.",
        "It constantly glitches and freezes.",
        "The battery life is horrendous and unacceptable.",
        "Way too expensive for such low quality.",
        "I will never purchase from this brand again.",
        "The smell is absolutely unbearable.",

        # --- Group 3: General Sentiment and Service (Positive) --- (Approx. 50 examples)
        "What a wonderful time we had!",
        "The entire team was courteous and professional.",
        "I feel so great about this decision.",
        "The delivery was prompt and on time.",
        "Everyone should try this restaurant.",
        "I'm feeling very optimistic about the future.",
        "This news makes me incredibly happy.",
        "A truly inspirational speech.",
        "The atmosphere was cozy and inviting.",
        "I had a lovely conversation with the representative.",
        "This trip was one of the best vacations ever.",
        "I appreciate the quick resolution to my issue.",
        "The food was delicious and perfectly prepared.",
        "That movie was a masterpiece, must watch.",
        "What a successful event! Everything went smoothly.",
        "My confidence level has increased dramatically.",
        "The sunlight streaming in is so beautiful.",
        "I am thankful for your help and support.",
        "This experience brightened my day considerably.",
        "I've never felt so energized and refreshed.",
        "The new policy is a brilliant idea.",
        "I am looking forward to the next meeting.",
        "A very positive step forward for the company.",
        "Their dedication to quality shines through.",
        "I am simply thrilled with the progress.",
        "The concert was absolutely electrifying.",
        "Such a positive and affirming message.",
        "Everything worked out beautifully in the end.",
        "I'm overjoyed by the excellent outcome.",
        "They deserve all the praise they get.",
        "This place has such good vibes.",
        "I'm totally satisfied with the information provided.",
        "The presentation was clear and insightful.",
        "It's a genuine pleasure to work with them.",
        "I love the attention to detail.",
        "A fantastic result for everyone involved.",
        "I am impressed by the speed of their service.",
        "This makes me incredibly hopeful.",
        "The service was fast, efficient, and friendly.",
        "What a great discovery this has been.",
        "I can finally relax and enjoy the moment.",
        "This new feature is super convenient.",
        "I am grateful for the chance to participate.",
        "It was a heartwarming scene.",
        "The organization was flawless.",
        "I'm feeling much better today.",
        "That's wonderful news!",
        "I applaud their commitment.",
        "This simplifies everything greatly.",
        "A truly remarkable achievement.",

        # --- Group 4: General Sentiment and Service (Negative) --- (Approx. 50 examples)
        "I'm very upset and angry about this situation.",
        "The service was incredibly slow and unprofessional.",
        "I had a dreadful time at the office.",
        "The waiting period was far too long.",
        "I feel completely frustrated and ignored.",
        "This issue has caused me a lot of stress.",
        "The communication was poor and misleading.",
        "I am absolutely furious with the delay.",
        "That movie was a disaster and a waste of time.",
        "The crowd was rude and unruly.",
        "I am highly skeptical of their claims.",
        "This is an unacceptable level of failure.",
        "I felt cheated by the hidden fees.",
        "The whole experience was disheartening.",
        "I will file a formal complaint immediately.",
        "Their lack of transparency is alarming.",
        "I'm struggling to understand the instructions.",
        "I have a deep sense of betrayal.",
        "The mistake was costly and avoidable.",
        "I really despise the new changes.",
        "This outcome is profoundly disappointing.",
        "I faced serious technical difficulties.",
        "The atmosphere was depressing and sterile.",
        "I was told conflicting information repeatedly.",
        "This leaves a very sour taste in my mouth.",
        "I am hesitant to trust this process again.",
        "The meeting was boring and pointless.",
        "I feel totally let down by the management.",
        "The food tasted bland and stale.",
        "It was a truly miserable failure.",
        "I'm concerned about the long-term viability.",
        "This has been a massive waste of resources.",
        "I don't think they handled the crisis well.",
        "The traffic was unbearable and caused chaos.",
        "I'm tired of the constant excuses.",
        "The error was obvious and should have been caught.",
        "I've never been so annoyed.",
        "Their policies are confusing and archaic.",
        "This whole ordeal has been exhausting.",
        "I see little reason for optimism.",
        "I have major reservations about this plan.",
        "The noise level was distracting and loud.",
        "I seriously doubt the accuracy of their data.",
        "The presentation lacked substance and depth.",
        "I feel completely alienated by the decision.",
        "This is an irresponsible use of funds.",
        "I was deeply offended by the comment.",
        "I hope this never happens again.",
        "The response was totally inadequate.",
        "I am burdened with extra work now.",

        # --- Group 5: Mixed/Nuanced Sentiment (Classified as 0 or 1 based on critical tone) --- (Approx. 50 examples)
        "It's okay, nothing special but it functions as advertised.", # 0 (Neutral, non-positive)
        "The design is nice, but the battery life is short and frustrating.", # 0 (Critical flaw)
        "I received the item quickly, but it was damaged in transit.", # 0 (Damage outweighs speed)
        "A decent effort, but there's significant room for improvement.", # 0 (Conditional, implies failure to meet expectations)
        "It did what it said it would, so I have no real complaints.", # 1 (Implied satisfaction)
        "I'm ambivalent about this purchase, it's just average.", # 0 (Neutral leaning)
        "The price was too high for the limited features offered.", # 0 (Critical of value)
        "It works mostly fine, but occasionally glitches or hangs.", # 0 (Glitches are a negative)
        "Not the best, but certainly not the absolute worst experience.", # 0 (Moderate negative/neutral)
        "The color is perfect, but the material feels a bit cheap.", # 0 (Flaw mentioned)
        "I would call it adequate; it gets the job done barely.", # 0
        "The initial setup was difficult, but the daily use is smooth.", # 1 (Focus on smooth use)
        "I had hoped for more innovation, but it's a solid product.", # 1 (Overall solid)
        "It's loud, but surprisingly effective at cleaning.", # 1 (Focus on effectiveness)
        "I am mostly happy, just wish the warranty was longer.", # 1 (Mostly happy)
        "It was acceptable for the low price point.", # 1 (Value-based positive)
        "The delivery was late, however, the product is excellent.", # 1 (Product quality is primary)
        "I can live with the minor flaw, the rest is superb.", # 1
        "The packaging was wasteful, though the contents were fine.", # 1
        "It's functional, yet lacks any real appeal.", # 0 (Lack of appeal is negative)
        "I found it mildly disappointing, to be honest.", # 0
        "They tried hard, but the result wasn't quite there.", # 0
        "I wouldn't praise it, but I wouldn't trash it either.", # 0
        "It's average at best; nothing to rave about.", # 0
        "I had to contact support, but they fixed the issue quickly.", # 1 (Quick fix is positive)
        "The presentation was long, but the information was valuable.", # 1
        "It was slow to start, but now it runs flawlessly.", # 1
        "The user interface is clunky, but the power is undeniable.", # 1
        "I got what I paid for; nothing more, nothing less.", # 1
        "The food was cold, but the server was very nice.", # 0 (Food is critical)
        "It’s almost perfect, if only the weight was lighter.", # 1 (Almost perfect)
        "It requires a software update, otherwise it's great.", # 1
        "The instructions were hard to follow, but I figured it out.", # 1
        "It seems reliable enough for casual use.", # 1
        "It's a step up from the old version, but still flawed.", # 0
        "I am not entirely satisfied with the outcome.", # 0
        "It was neither good nor bad, just utterly forgettable.", # 0
        "I'm indifferent to the design changes.", # 0
        "The initial impression was poor, but it grew on me.", # 1
        "I had to wait a long time, but the end result was worth it.", # 1
        "I regret not spending a little more on a premium option.", # 0
        "It's too complicated for the average user.", # 0
        "The concept is fantastic, the execution is weak.", # 0
        "It's a serviceable item, nothing more.", # 1
        "I am a little confused by the setup, but I'll manage.", # 0
        "The staff seemed stressed, but they were efficient.", # 1
        "I wouldn't say I love it, but I don't dislike it either.", # 0
        "The sound quality is amazing, but the controls are fiddly.", # 0
        "It's reasonably priced and performs adequately.", # 1
        "I'm marginally satisfied with this version." # 1
    ]

    # The labels are compiled based on the overall dominant sentiment of each sentence.
    labels = np.array([
        # Group 1: Product Positive (25)
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        # Group 2: Product Negative (25)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        # Group 3: General Positive (50)
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        # Group 4: General Negative (50)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        # Group 5: Mixed/Nuanced (50)
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1,
        1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1
    ])

    return sentences, labels

def load_dataset_initial():
    # toy dataset; replace with real data loader in src/data
    sentences = [
        "I love this product, it works great",
        "Absolutely fantastic service",
        "This is the best I've used",
        "Very satisfied and happy",
        "I hate this, it broke immediately",
        "Terrible experience, would not recommend",
        "Awful, I am disappointed",
        "Not good, very poor quality"
    ]
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    return sentences, labels