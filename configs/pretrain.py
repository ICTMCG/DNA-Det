class Config(object): 
    # random seed
    seed = 0

    # optimize
    init_lr_E = 1e-4
    step_size = 500
    gamma = 0.9

    # dataset
    batch_size = 128
    num_workers = 4
    class_num = 170
    crop_size = (128,128)
    resize_size = (512,512)
    second_resize_size = None
    multi_size = [(64,64)]*16
    
    # loss
    temperature = 0.07
    
    # model_selection
    metric = 'f1'
    max_epochs = 50
    early_stop_bar = 20
    save_interval= 5

