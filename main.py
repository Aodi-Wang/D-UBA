from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_trainer, init_seed, set_color

if __name__ == '__main__':
    # configurations initialization
    config = Config(
        model=RceIAN,
        dataset='retailrocket',  # diginetica ; retailrocket ; tmall
        config_file_list=['config.yaml', 'config_model.yaml'],
    )
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = D-UBA(config, train_data.dataset).to(config['device'])



    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data,
    )


