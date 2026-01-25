from schnapsen.game import Bot,SchnapsenGamePlayEngine
from schnapsen.bots import  RandBot , MLPlayingBot,MLDataBot,train_ML_model
from pathlib import Path
from random import Random
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

def iterative_training(
        training_opponents: list,
        total_games: int,
        rng: Random, 
        model_class: str="LR",
        iterations: int=10
    ) -> tuple[dict[str, list[Bot]], list[Bot]]:
    """
    Function for training the models and saving their performance progress at set intervals.

    Args:
        training_opponents: list - (Quality test) A list of opponents used in the experiment (
        RandBot - "Low" level data
        BullyBot - "Medium" data
        Rdeep - "High" data | The level can wary between the depth and sample amount of the RdeepBot (Can be expanded in future testing)
        Mixed - "Highend_Medium/ Lowend_High"? data | Changing opponents can confuse our ML model and it might not develop a good strategy?
        ________________________________________________________________________________________________________________________________________________________________________
        total_games: int - (Quantity test) Total game amount the ML will be trained on
        rng: Random - Seed used in experiment, for tracking results and ensuring the results are able to be reproduced
        model_class: str - Selected way of training the model (LR|NN) (default = LR)
        iterations: int - How many points we want to observe the bots progress in (Used in plotting the results when making learning curve) (default = 10)
        If we used "Mixed" training (Mixed plays less games against each opponent but the same amount of games in total)

        H1: mixed training leads to to lower overall performace but better genaralisation
        H0: mixed training is AMAZING O_O bot perfomance is the best overall   
    
    Returns:
        tuple[single_models_by_opponent: dict[str, list[Bot]], mixed_models: list[Bot]]:
        
    """
    games_per_iteration = total_games // iterations

    #Directory setup
    # Trained against a single opponent
    replay_dir = ROOT_DIR / f"{model_class}_ML_replay_memories"
    model_dir = ROOT_DIR / f"{model_class}_ML_models"

    # Trained against mixed opponents
    mixed_replay_dir = ROOT_DIR / f"Mixed_{model_class}_ML_replay_memories"
    mixed_model_dir = ROOT_DIR / f"Mixed_{model_class}_ML_models"
    
    replay_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True,exist_ok=True)
    mixed_replay_dir.mkdir(parents=True, exist_ok=True)
    mixed_model_dir.mkdir(parents=True,exist_ok=True)

    # One opponent training
    single_models_by_opponent = train_single_opponent_models(training_opponents,games_per_iteration,iterations,total_games,replay_dir,model_dir,model_class,rng)
    # Mixed training
    mixed_models = train_mixed_opponent_models(training_opponents,games_per_iteration,iterations,total_games,mixed_replay_dir,mixed_model_dir,model_class,rng)

    return single_models_by_opponent, mixed_models

def train_single_opponent_models(  
        training_opponents: list[Bot],
        games_per_iteration: int,
        iterations: int,
        total_games: int,
        replay_dir: Path,
        model_dir: Path,
        model_class: str,
        rng: Random
        ) -> dict[str, list[Bot]]:
    """
    Function For training a ML model against one single opponent for x amount of games

    Args:
        training_opponents: list[Bot] - (Quality test) A list of opponents used in the experiment
        games_per_iteration: int - Amount of games to use to train a specific iteration of the model
        iterations: int - Amount of total model iterations
        total_games: int - The total games the models will be trained on
        replay_dir: Path - folder where to save the model replays
        model_dir: Path - folder to where to save the models
        model_class: str - Selected way of training the model (LR|NN)  (default = LR)
        rng: Random - Set seed used in creating the models for future reproduction

    Return:
        single_models_by_opponent: dict[str, list[Bot]] - A dict of bots where key is bot trained against item is the List of models
    """
    
    single_models_by_opponent = {}
    for opponent in training_opponents:
        single_models_by_opponent[opponent] = []
        behaviour_bot = RandBot(rng,"Randy")
        games_seen = 0

        replay_file = replay_dir / f"replay_{opponent}_{total_games}_{model_class}.txt"
        if replay_file.exists():
            replay_file.unlink()

        for iteration in range(iterations): # Iteration training process loop
            games_seen +=games_per_iteration
            model_file =  model_dir / f"{opponent}_{model_class}_iter_{iteration}_games{games_seen}_from{total_games}.model"

            trained_model_path = updating_ml_bot(behaviour_bot, opponent, games_per_iteration, replay_file, model_file, model_class, rng)

            if trained_model_path is not None:
                evaluation_bot = MLPlayingBot(model_location=model_file,name=f"{opponent}_{model_class}_iter_{iteration}_games{games_seen}_from{total_games}")
                behaviour_bot = evaluation_bot
                single_models_by_opponent[opponent].append(evaluation_bot)
            else:
                raise Exception(f"Training failed at iteration {iteration}")
    return single_models_by_opponent
            
def train_mixed_opponent_models(
        training_opponents: list[Bot],
        games_per_iteration: int,
        iterations: int,
        total_games: int,
        replay_dir: Path,
        model_dir: Path,
        model_class: str,
        rng: Random
        ) -> list[Bot]:
    """
    Function For training a ML model against mixed opponent for x amount of games
    Args:
        training_opponents: list[Bot] - (Quality test) A list of opponents used in the experiment
        games_per_iteration: int - Amount of games to use to train a specific iteration of the model
        iterations: int - Amount of total model iterations
        total_games: int - The total games the models will be trained on
        replay_dir: Path - folder where to save the model replays
        model_dir: Path - folder to where to save the models
        model_class: str - Selected way of training the model (LR|NN)  (default = LR)
        rng: Random - Set seed used in creating the models for future reproduction
    Return:
        mixed_models: list[Bot] - A list of bots after they have been trained 
    """
    behaviour_bot = RandBot(rng,"Randy")
    games_seen = 0
    mixed_models = []

    replay_file = replay_dir / f"replay_Mixed_{total_games}_{model_class}.txt"
    if replay_file.exists():
        replay_file.unlink()

    for iteration in range(iterations): # Iteration training process loop
        opponent = training_opponents[iteration % len(training_opponents)]
        games_seen +=games_per_iteration
        model_file =  model_dir / f"Mixed_{model_class}_iter_{iteration}_games{games_seen}_from{total_games}.model"

        trained_model_path = updating_ml_bot(behaviour_bot, opponent, games_per_iteration, replay_file, model_file, model_class, rng)

        if trained_model_path is not None:               
            evaluation_bot = MLPlayingBot(model_location=model_file,name=f"Mixed_{model_class}_iter_{iteration}_games{games_seen}_from{total_games}.model")
            behaviour_bot = evaluation_bot
            mixed_models.append(evaluation_bot)

        else:
            raise Exception(f"Training failed at iteration {iteration}")
    return mixed_models
            
def updating_ml_bot(
        behaviour_ml_bot: Bot,
        opponent: Bot,
        games_to_play: int,
        replay_file: Path,
        model_file: Path,
        model_class: str,
        rng: Random
        ) -> Path:
    """
    Plays games against a fixed opponent, stores replay data, and trains an ML model.

    Args:
        behaviour_ml_bot: Bot - Current policy used to generate training data.
        opponent : Bot - Opponent played during this training phase.
        games_to_play : int - Number of games to play in this iteration.
        replay_file : Path - Replay memory file (appended across iterations).
        model_file : Path - Output path for the trained model.
        model_class : str - ML model type ("LR" | "NN").
        rng : Random - RNG for reproducibility.

    Returns:
        Path|None -Path to trained model if successful, otherwise None.
    """
    eng = SchnapsenGamePlayEngine()
    model_file.parent.mkdir(parents=True, exist_ok=True)
    replay_file.parent.mkdir(parents=True, exist_ok=True)
    ml_bot = MLDataBot(bot=behaviour_ml_bot, replay_memory_location=replay_file)

    for game_index in range(games_to_play): # Play games for training data
        eng.play_game(ml_bot,opponent,Random(rng.randint(0,1000000)))

    if check_replay_file(replay_file):
        train_ML_model(replay_memory_location=replay_file, model_location=model_file, model_class=model_class)
        return model_file
    else:
        print(f"Insufficient data in replay file {replay_file} vs {opponent}, skipping model training.")
        return None

def check_replay_file(replay_file: Path) -> bool:
    """
    Check that the replay file countains multiple outcomes (Wins|Loses) to make sure training is possible.

    Args:
        replay_file: Path - The location of the replay file

    Returns:
        bool: (True|False)   
    """
    if not replay_file.exists():
        return False

    labels = set() #Only check whether at least two outcome classes exist.
    with open(replay_file, "r") as f:
        for line in f:
            if "||" not in line:
                continue
            _, label = line.strip().split("||")
            labels.add(label)
            if len(labels) >= 2:
                return True
    return False
