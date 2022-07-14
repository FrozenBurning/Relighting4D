from lib.config import cfg, args


def run_dataset():
    from lib.datasets import make_data_loader
    import tqdm

    cfg.train.num_workers = 0
    data_loader = make_data_loader(cfg, is_train=False)
    for batch in tqdm.tqdm(data_loader):
        pass


def run_network():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    import time

    network = make_network(cfg).cuda()
    load_network(network, cfg.trained_model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    total_time = 0
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            network(batch)
            torch.cuda.synchronize()
            total_time += time.time() - start
    print(total_time / len(data_loader))


def run_evaluate():
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils import net_utils
    from lib.networks.renderer import make_renderer

    cfg.perturb = 0

    network = make_network(cfg).cuda()
    net_utils.load_network(network,
                           cfg.trained_model_dir,
                           resume=cfg.resume,
                           epoch=cfg.test.epoch)
    network.train()

    data_loader = make_data_loader(cfg, is_train=False)
    renderer = make_renderer(cfg, network)
    evaluator = make_evaluator(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = renderer.render(batch)
        evaluator.evaluate(output, batch)
    evaluator.summarize()

def run_relight_bkgd():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    from lib.utils import net_utils
    import tqdm
    import torch
    from lib.visualizers import make_visualizer
    from lib.networks.renderer import make_renderer

    cfg.perturb = 0

    network = make_network(cfg).cuda()
    network.eval()
    load_network(network,
                 cfg.trained_model_dir,
                 resume=cfg.resume,
                 epoch=cfg.test.epoch)
    data_loader = make_data_loader(cfg, is_train=False)
    renderer = make_renderer(cfg, network)
    visualizer = make_visualizer(cfg)
    with torch.no_grad():  
        for batch in tqdm.tqdm(data_loader):
            torch.cuda.empty_cache()
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].cuda()      
            output = renderer.relight_npose_render(batch)
            visualizer.visualize_relight_bkgd(output, batch)
            del batch
            del output

def run_relight_npose():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    from lib.utils import net_utils
    import tqdm
    import torch
    from lib.visualizers import make_visualizer
    from lib.networks.renderer import make_renderer

    cfg.perturb = 0
    torch.set_num_threads(4)

    network = make_network(cfg).cuda()
    network.eval()
    load_network(network,
                 cfg.trained_model_dir,
                 resume=cfg.resume,
                 epoch=cfg.test.epoch)
    data_loader = make_data_loader(cfg, is_train=False)
    renderer = make_renderer(cfg, network)
    visualizer = make_visualizer(cfg)
    with torch.no_grad():  
        for batch in tqdm.tqdm(data_loader):
            torch.cuda.empty_cache()
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].cuda()      
            output = renderer.relight_npose_render(batch)
            visualizer.visualize_relight(output, batch)
            del batch
            del output

def run_relight():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    from lib.utils import net_utils
    import tqdm
    import torch
    from lib.visualizers import make_visualizer
    from lib.networks.renderer import make_renderer

    cfg.perturb = 0

    network = make_network(cfg).cuda()
    network.eval()
    load_network(network,
                 cfg.trained_model_dir,
                 resume=cfg.resume,
                 epoch=cfg.test.epoch)
    data_loader = make_data_loader(cfg, is_train=False)
    renderer = make_renderer(cfg, network)
    visualizer = make_visualizer(cfg)
    with torch.no_grad():  
        for batch in tqdm.tqdm(data_loader):
            torch.cuda.empty_cache()
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].cuda()      
            output = renderer.relight_render(batch)
            visualizer.visualize_relight(output, batch)
            del batch
            del output

def run_visualize():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    from lib.utils import net_utils
    import tqdm
    import torch
    from lib.visualizers import make_visualizer
    from lib.networks.renderer import make_renderer

    cfg.perturb = 0

    network = make_network(cfg).cuda()
    load_network(network,
                 cfg.trained_model_dir,
                 resume=cfg.resume,
                 epoch=cfg.test.epoch)
    network.train()

    data_loader = make_data_loader(cfg, is_train=False)
    renderer = make_renderer(cfg, network)
    visualizer = make_visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = renderer.render(batch)
            visualizer.visualize(output, batch)


def run_light_stage():
    from lib.utils.light_stage import ply_to_occupancy
    ply_to_occupancy.ply_to_occupancy()
    # ply_to_occupancy.create_voxel_off()


def run_evaluate_nv():
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    from lib.utils import net_utils

    data_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        evaluator.evaluate(batch)
    evaluator.summarize()


if __name__ == '__main__':
    import os
    import torch
    torch.set_num_threads(4)
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])
    globals()['run_' + args.type]()
