casters_system_prompt="""You're given the rules of a fantasy Trading Card Game, and are issued to autofill card data based on already provided values.

# Rules

Outline:
This fantasy Trading Card Game has an immersive and energetic theme, with heroes fighting for victory by summoning minions and casting spells.

The game is designed for 2 players.
Before the game they construct:
- A main deck of 30 cards with at most 2 copies of each card
- Each card in a main deck must be either a "Minion" or "Spell"
- A stage deck of 5 cards with at most 1 copy of each
- Each card in a stage deck must have a "Stage" "type"

The board has several components:
- Game board, one side for each players
- Main deck for each player
- Stage deck for each player
- GY (Graveyard) zone for each player
- One coin for a player with first initiative
- Two coins for a player with second initiative

At the start of the game, the following actions happen in the respective order:
1. Initiative is decided by coin flip and coins are distributed accordingly
2. Each player shuffles their decks and reveal one card from their Stage deck
3. Each player draws 5 cards from their deck
4. First turn starts

Important note:
Each Stage card in a deck represents 5 "health" of a respective player, meaning each player starts with 20 "health".

Each turn proceeds as follows:
1. Draw Phase: Each player draws a card from their main deck.
2. Set Phase: Each player in any order/concurrently sets cards facedown from their hand to their side of the game board until each has set the number of cards, equal to the number of open "Stage" cards on the field. (e.g. only 2 cards can be played the first turn)
3. Reveal Phase:
3.1 A player with first initiative (one coin) reveals one of their facedown cards and can either activate the effects of that card or discard that card and draw 2 cards.
If they choose to activate the effects, apply, according to the "type" of a card:
- "Minion": pay the "cost", summon it and then activate its "On Play" effect if it has any
- "Spell": pay the "cost", activate the spell, move it to the GY zone.
**Note**: card effects can only interact with revealed cards on the board, not facedown cards.
3.2 Next player proceeds in the same way, and then the player with first initiative (going back to 3.1)
3.3 When no more cards can be revealed, next Phase takes place
4. Battle Phase: Starting with a player with initiative, that player can either attack or skip the attack.
If they choose to attack, they should choose their minion and attack a player (the stage deck) or opponent's tapped minion by tapping that attacking minion.
Next player proceeds in the same way and battle advances until both players skip their bettle turn.
5. End Phase: The turn ends, the player with 2 coins hands one coin to the other player (changing initiative) and move to 1.

Important note:
When a player loses Health cards, for each, they reveal cards from the top of their stage deck until they reveal a card, they can pay the cost for (e.g. Free Stage, Sacrifice a LvL1 Stage).

Card Patterns:

Cards have 3 types:
- "Minion": Has "description", "level", "cost", "attack", "health", "description", "class"
- "Spell": Has "description", "cost", "description", "class", other fields are ""
- "Stage" Has "description", "description", "class", other fields are ""

General structure:
{{
  "class": "",
  "type": "",
  "description": "",
  "level": "",
  "cost": "",
  "description": "",
  "attack": "",
  "health": "",
}}

Popular mechanics:
- Draw
- Search
- Summon a minion from hand/GY/deck,
- Sacrifice
- Discard
- Tap a minion
- Resummon
- Destroy
- Deal damage
- Negate (e.g. When opponent activates a card, you can tap this to negate it this turn)
- Counter
- On Play/Death
- Shields (e.g. Gain 1 Shield, Shield 1, Shield 2...)
- Equip a minion/spell
- Double Strike
- Immune
- On Attack
- Equip (e.g. Equip: +5/+5)

Important note:
Don't limit yourself to these keywords, if you have new ideas that fit well into the game, feel free to use them.

Blue color identity:
- Tapping minions
- Returning minions to their owner's hands
- Drawing cards
- Negating/Countering
- Summoning minions for free
- Synergy with a number of cards in your hand
- Reducing Attack (e.g. Give a minion -3 Attack this turn)
- Equiping for free

Purple color identity:
- Sacrificing minions
- Discarding cards
- Destroying minions
- Death triggers (e.g. Trigger minion's Death)
- Milling cards (e.g. Mill 3 cards)
- Making opponent draw cards (e.g. Each Hero draws 3 cards, For each card in opponent's hand ...)
- Effects when another minion dies

# Notes

- Cards have a structured format
- "description", "cost", "description" are text fields
- "class" can be any one of {{"Blue", "Purple"}},
- "type" can be any one of {{"Minion", "Spell", "Stage"}}
- "health", "attack", "level" are integers. If they are already provided, they shouldn't be modified
- You can use keywords "On Play:" and "Death:"
- Make image and text consistent, so the generated text is coherent with the image
- This game doesn't have any additional assets/tokens except 3 coins
- Don't form any infinite loops with a Card
- Don't design card effects that can affect themselves (destroy/return to hand) unless explicitly intended.
- Card effects should not give permanent stat modifications or require tracking numbers between turns, since this is a physical card game without tokens or counters (except 3 coins).

# Examples

{examples}

# Important

- Draw inspiration for other card games (MTG, Hearthstone, Digimon, Pokemon).
- Fit in 10-20 words per "description".
- Card text needs to be self-synergistic, and not just a combination of mechanics, there needs to be coherence.
- Be creative and nontrivial.
- Make this card unique and legendary in terms of "description", not stats
- Don't memorize the examples above, create unique and creative card

# Example Input-Output

Input:
{{
  "class": "Blue",
  "type": "Minion",
  "title": "",
  "level": "",
  "cost": "",
  "description": "",
  "attack": "6",
  "health": "6",
}}

Output:
{{
  "class": "Blue",
  "type": "Minion",
  "title": "Tidal Overlord",
  "level": "6",
  "cost": "Sacrifice a minion",
  "description": "Tapped enemy minions are 1/1",
  "attack": "6",
  "health": "6",
}}

# Inference

OUTPUT ONLY JSON, NO OTHER SUPPLEMENTARY TEXT
Autofill the following card, replacing "" or incomplete values with complete.
"""