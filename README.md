# GPT
## GPT Paper Summaries
* [GPT](https://colab.research.google.com/drive/1d4BmKVoNGREQR2j2yv9lHORrcWS4eLgl#scrollTo=AP2x1jC9-319)
* [GPT-2](https://colab.research.google.com/drive/1d4BmKVoNGREQR2j2yv9lHORrcWS4eLgl#scrollTo=yHOofcd8Jajj)
* [GPT-3](https://colab.research.google.com/drive/1d4BmKVoNGREQR2j2yv9lHORrcWS4eLgl#scrollTo=mlHE3Xmjo290)
* GPT-4: Multimodal Model, not open-sourced by OpenAi :(
* Also check out my Transformer repo [Attention-Is-All-You-Need](https://github.com/VachanVY/Attention-Is-All-You-Need)
* Check out the `.ipynb` files for full training guide

## Trained on [Shakespeare Dataset](https://homl.info/shakespeare):
### Results
* Validation Loss of **1.4790** with a character-level model (6 Million parameters)

### Predictions
* Sampling technique: `Random Categorical`; `temperature=0.80`
```
RICHARD:
Shall I meet thee on you should pluck a blame
Of the duke will go wonder.

CORIOLANUS:
Come, cousin!

HERMIONE:
Brother, better than bitter, and you are rule in his
terrorious character, therefore shall after for him,
Many eyes of this ancient to his country:
And ere they scarce to possession,
As from me thy blood master good will you once ear.

GLOUCESTER:
Thou shalt do it please your grace.

CLARENCE:
What's the royal beauty?

ROMEO:
Hark you not, let me speak to princely out of any.

QUEEN ELIZABETH:
Well, well met them.

DUKE VINCENTIO:
A pity, a little prince in them, but I say, then?
O woman, sir, I am a charelessed was that
not deserved with limits that so in my mind
Than she shall live to follow these true just
To be worthy talked of our cause, her chains
That is the palace of men. I pray you, sir.

MERCUTIO:
O Play the Duke of God!

Provost:
I shall be proof. What shall you have been true?

MISTRESS OVERDONE:
I'll sure thee to see the divine.

KING RICHARD III:
Hath sends him here will speak to bear the middle.

Nurse:
I am so much as like a store than my prosperous
tongue in the people's knees. Have I repent thee
Against the love with him up to do it.
...
```

* Sampling technique: `top-k=5`; `temperature=0.80`
```
KING EDWARD IV:
But what's the princes of me, and they shall pardon thee,
And, as the part in all power of this deeds
Of my sweet boys before I cannot do't;
And so her fair a person of your chamber,
The presently gods have more private to see
The crown another did lurk to she should be sentence.

LUCIO:
And thou art pluck'd as the sweetest subjects.

DUKE VINCENTIO:
Thou'rt be the case of teach him.

Second Citizen:
I should say 'twas to my lips.

CORIOLANUS:
I do remain to meet the people.

Provost:
Madam, we have no more. That is a character's child?

PAULINA:
I cannot damnable and honesty.

LUCIO:
A most graciously since I will not be so,
If you shall be married with the sun,
The trumpets, and the stocks of his place.

KING HENRY VI:
Well, sir, the gods but one of your honours.

MERCUTIO:
I think thee a party times and seen them,
And where you did be a chustic and safety thee,
To make thee seek to be the wrong.

PETRUCHIO:
An is in the morning of him well in her
Beauties the chamber, who is the chair,
Which should show me to my head?

LORD ROSS:
Ay, my lord, my master was all most promise.
...
```

* Greedy Sampling
* This sampling technique isn't good. Bad Predictions
```
I the shoon ther the
shout to bearry.

KING RICHARD III:
What is the strampedies
Aue thantie tend the air oful day.

KING RICHARD IORD LARDOHARDARD IINRE:
Was nothing bless at ble ble bombombras,
Aneft the sea of thearrmonithe ale mottte,
amoon'sh thatt the sea of the sea,
And therefore are the season of the sea,
And therefore are the season of the sea,
And therefore are the season of the sea,
And therefore are the season of the sea,
And therefore are the season of the sea,
And therefore are the ...
```
