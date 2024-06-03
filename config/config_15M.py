from dataclasses import dataclass
from typing import Callable
import keras as nn

@dataclass
class GPTConfig:
    """GPT 15M Configuration"""
    use_flash_att:bool=True
    d_model:int = 288
    num_layers:int = 6
    num_heads:int = 6
    maxlen:int = 256
    vocab_size:int = 32_000
    output_units:int = None
    assert d_model % 2 == 0
    assert d_model % num_heads == 0
    dropout_rate:float = 0.0
    use_bias:bool = True
    intializer:Callable = lambda std: nn.initializers.RandomNormal(mean=0.0, stddev=std)


# Random Seed: 4950
# Once upon a time, there was a small dog named Spot. Spot loved to play with his ball. One day, Spot went to the park with his best friend, a little girl named Lily.
# At the park, Spot and Lily played catch with the ball. They ran and laughed, having lots of fun. But then, something unexpected happened. A big dog came running and took the ball away. Spot and Lily were sad.
# But then, the big dog came back with the ball. The dog dropped the ball and started to play with them. Spot and Lily were so happy! They all played together and became best friends. The big dog was their new friend, and they all had a great day at the park.

# Time Taken (sec): 391.9155411720276
# Tokens per second: 0.4056996554015409 


# Random Seed: 3301
# Once upon a time, there was a mighty dog named Max. Max had a job to deliver things to his friends. He had a job to deliver toys to everyone in the town. Max was a very good dog, and he always made sure everyone was happy.
# One day, Max met a little bird who was lost. The bird said, "I can't find my way home. Can you help me?" Max said, "Yes, I will help you." So, Max and the bird started to look for the bird's home.
# They walked and walked, but they could not find the bird's home. Then, they saw a big cat. The cat said, "I know where your home is! Follow me!" Max and the bird followed the cat. They found the bird's home and the bird was very happy. Max's job was a success, and he felt good about helping his new friend.

# Time Taken (sec): 72.79522109031677
# Tokens per second: 2.6650101077281554 


# Random Seed: 7157
# Once upon a time, there was a little boy named Tim. He had a toy car that he loved very much. The toy car had a shiny buckle on it. Tim took it everywhere he went.
# One day, Tim went to the park with his mom. He brought his shiny buckle with him. At the park, Tim played with his friends and had lots of fun. But then, Tim started to worry. He thought his shiny buckle was too shiny and he didn't want to play with his friends anymore.
# Tim's mom saw that he was worried and asked him what was wrong. Tim told her about his shiny buckle. His mom smiled and said, "Don't worry, Tim. Your buckle is just as shiny as your buckle. Your friends can help you with your shiny buckle, and they can make it shiny." Tim felt better and went back to play with his friends. And they all played happily ever after.

# Time Taken (sec): 53.35686159133911
# Tokens per second: 4.10443930674416 


# Random Seed: 6386
# Once upon a time, there was a little girl named Lucy. She had a big box of junk in her room. She liked to play with the junk and make things with it. One day, she saw a small, helpless cat outside her window.
# Lucy wanted to help the cat. She took the cat inside her house. She gave it some food and water. The cat was happy to have a home. Lucy felt good because she was kind and helpful.
# One day, Lucy's friend came to play. They played with the junk and the cat. Then, something unexpected happened. The cat started to talk! The cat said, "Thank you for being so kind." Lucy and her friend were surprised but happy. They had a new friend to play with and help each other.

# Time Taken (sec): 1.5343642234802246
# Tokens per second: 109.49160403319665 


# Random Seed: 7891
# Once upon a time, there was a little girl named Mia. Mia had a big book called a dictionary. She loved to learn new words in her dictionary. One day, Mia's friend Tom came over to play.
# Mia said, "Tom, I want to show you my dictionary. It is very big and impressive." Tom looked at the dictionary and said, "Wow, it is so big! Can you help me open it?" Mia smiled and said, "Yes, I can help you."
# Mia and Tom opened the dictionary together. Inside, they saw many colorful words. They took out the words and started to read them. Mia and Tom were very happy. They learned many new words and had lots of fun.

# Time Taken (sec): 0.8383712768554688
# Tokens per second: 190.84623294839244 


# Random Seed: 9464
# Once upon a time, there was a little girl named Mia. Mia loved to play outside with her friends. One day, Mia's mom said, "Mia, I will introduce you to a new friend." Mia was very happy and excited.
# Mia's mom took her to a big park. There were many kids playing and having fun. Mia saw a big tree and wanted to play near it. She saw a big swing and ran to it. Mia's mom said, "Wait, Mia! The swing is available for you to play on."
# Mia sat on the swing and her friends watched her. They all took turns swinging and laughing. Mia felt so happy that she could swing high and have fun with her friends. And they all played together until the sun went down.

# Time Taken (sec): 0.8343756198883057
# Tokens per second: 210.93617287566525 


# Random Seed: 9441
# Once upon a time, there was a young boy named Tim. He lived in a small house with his mom and dad. One day, Tim saw a big tree near his house. He wanted to climb the tree to see the view from the top.
# As Tim climbed the tree, he saw a little bird. The bird said, "Hi, Tim! What are you doing?" Tim replied, "I want to see the view from the top of the tree!" The bird smiled and said, "Be careful, Tim! The view is very high up here."
# Tim listened to the bird and decided to help the bird. He climbed down the tree and told his mom and dad about the view. They were happy that Tim was so kind and helpful. From that day on, Tim always helped his friends when they needed him.

# Time Taken (sec): 1.0612916946411133
# Tokens per second: 164.89340384330254 


# Random Seed: 6462
# Once upon a time, there was a little boy named Tim. Tim had a toy car that he loved very much. One day, the toy car went missing, and Tim was very sad.
# Tim's mom saw that he was sad and asked, "Tim, why are you sad?" Tim said, "Mom, I lost my toy car. Can you help me find it?" His mom said, "Yes, let's look for it together."
# They looked all over the house, but they could not find the toy car. Tim was very sad. Then, Tim's mom had an idea. She said, "Let's replace the toy car with a new one." Tim's face lit up with a big smile.
# Together, they found a new toy car for Tim. Tim was very happy, and he said, "Thank you, Mom!" They played with the new toy car and had lots of fun. Tim learned that when you feel sad, you can always find a way to help your friends.

# Time Taken (sec): 9.196376323699951
# Tokens per second: 24.24868145350984 


# Random Seed: 9062
# Once upon a time, in a small village, there was a little girl named Mia. Mia was a fearful girl who loved to play outside. One sunny day, Mia went to the park to play.
# At the park, Mia saw a big tree. She wanted to climb it, but she was scared. Mia's friend, Tom, came to the park too. Tom said, "Don't be scared, Mia. I am here with you."
# Mia and Tom climbed the big tree together. They held on tight to the tree. When they reached the top, they saw a big nest. Inside the nest, there were three baby birds. The baby birds were hungry and scared.
# Mia and Tom decided to help the baby birds. They gave them food and water. The baby birds were happy and became Mia and Tom. They played together in the park every day. Mia was not fearful anymore.

# Time Taken (sec): 0.6686201095581055
# Tokens per second: 308.0972245003915 


# Random Seed: 4963
# Once upon a time, there was a little boy named Tim. Tim had a favorite seat in his house. It was a big, soft, and comfortable seat. He loved to sit on it and read books. One day, Tim saw a small, shiny ball on the floor. He wanted to play with it.
# Tim picked up the ball and gave it a big squeeze. The ball started to roll and bounce. Tim was so happy to see the ball roll. He squeezed the ball again and again. The ball was so much fun to play with.
# Then, something unexpected happened. The ball started to grow bigger and bigger! Tim was very surprised. The big ball was not a ball at all, but a big, soft pillow! Tim laughed and took a nap on the big, soft pillow. He had a fun day playing with the big, soft pillow.

# Time Taken (sec): 0.6525721549987793
# Tokens per second: 289.62314519894517
