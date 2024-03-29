import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
from ocvrp import algorithms
from ocvrp.cvrp import CVRP
from ocvrp.util import CVRPEncoder


def pos_float(value):
    try:
        value = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError('Value must be numerical')

    if value < 0 or value > 1:
        raise argparse.ArgumentTypeError('Value must be >= 0 and <= 1')

    return value


def pos_int(value):
    try:
        value = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('Value must be an integer')

    if value < 0:
        raise argparse.ArgumentTypeError('Value must be >= 0')
    return value


def int_ge_one(value):
    try:
        value = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('Value must be an integer')

    if value <= 0:
        raise argparse.ArgumentTypeError('Value must be >= 1')
    return value


class ValidOutputFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            Path(values).mkdir(parents=True)
        except IOError as ioe:
            if hasattr(ioe, 'message'):
                argparse.ArgumentTypeError(ioe.message)
            else:
                argparse.ArgumentTypeError(ioe)
        setattr(namespace, self.dest, values)


def main():
    pop = 800
    sel = 5
    ngen = 100_000
    mutpb = 0.15
    cxpb = 0.85
    cx_algo = algorithms.best_route_xo
    mt_algo = algorithms.inversion_mut

    parser = argparse.ArgumentParser(description="Runs the CVRP with any of the optional arguments")
    parser.add_argument("directory", type=str, help="the directory containing problem set files")
    parser.add_argument("-o", "--output", action=ValidOutputFile, metavar='', type=str,
                        help="the path to output the results (creates the path if it does not exist)")
    parser.add_argument("-p", "--pop", metavar='', type=pos_int, help="the population size")
    parser.add_argument("-s", "--sel", metavar='', type=pos_int, help="the selection size")
    parser.add_argument("-g", "--ngen", metavar='', type=pos_int, help="the generation size")
    parser.add_argument("-m", "--mutpb", metavar='', type=pos_float, help="the mutation probability")
    parser.add_argument("-c", "--cxpb", metavar='', type=pos_float, help="the crossover probability")
    parser.add_argument("-r", "--run", metavar='', type=int_ge_one, help="the number of times to run the problem")

    cx_types = parser.add_mutually_exclusive_group()
    cx_types.add_argument("-B", "--brxo", action='store_true', help="use best route crossover")
    cx_types.add_argument("-C", "--cxo", action='store_true', help="use cycle crossover")
    cx_types.add_argument("-E", "--erxo", action='store_true', help="use edge recombination crossover")
    cx_types.add_argument("-O", "--oxo", action='store_true', help="use order crossover")

    mt_types = parser.add_mutually_exclusive_group()
    mt_types.add_argument("-I", "--vmt", action='store_true', help="use inversion mutation")
    mt_types.add_argument("-W", "--swmt", action='store_true', help="use swap mutation")
    mt_types.add_argument("-G", "--gvmt", action='store_true', help="use GVR based scramble mutation")

    parser.add_argument("-i", "--indent", metavar='', nargs="?", type=int_ge_one, const=2,
                        help="the indentation amount of the result string")
    parser.add_argument("-P", "--pgen", action='store_true', help="prints the current generation")
    parser.add_argument("-A", "--agen", action='store_true', help="prints the average fitness every 1000 generations")

    parser.add_argument("-S", "--save", action="store_true", help="saves the results to a file")
    parser.add_argument("-R", "--routes", action="store_true", help="adds every route (verbose) of the best "
                                                                    "individual to the result")
    parser.add_argument("-M", "--plot", action="store_true", help="plot average fitness across generations with "
                                                                  "matplotlib")
    args = parser.parse_args()

    directory = args.directory
    if not os.path.isdir(directory):
        print("Error: The specified directory does not exist.")
        return

    output = args.output if args.output else "./results"
    if not os.path.isdir(output):
        os.mkdir(output)
        
    output_data = {}
    for file_name in tqdm(os.listdir(directory)):
        if file_name.endswith(".ocvrp"):
            file_path = os.path.join(directory, file_name)
            print(f"Processing file: {file_path}")

            p_set = file_path

            pop = args.pop if args.pop else pop
            sel = args.sel if args.sel else sel
            ngen = args.ngen if args.ngen else ngen
            mutpb = args.mutpb if args.mutpb else mutpb
            cxpb = args.cxpb if args.cxpb else cxpb

            runtime = args.run if args.run else 1
            if args.cxo:
                cx_algo = algorithms.cycle_xo
            elif args.erxo:
                cx_algo = algorithms.edge_recomb_xo
            elif args.oxo:
                cx_algo = algorithms.order_xo

            if args.swmt:
                mt_algo = algorithms.swap_mut
            elif args.gvmt:
                mt_algo = algorithms.gvr_scramble_mut

            runs = {"RUNS": {}}

            cvrp = CVRP(problem_set_path=p_set,
                        population_size=pop,
                        selection_size=sel,
                        ngen=ngen,
                        mutpb=mutpb,
                        cxpb=cxpb,
                        pgen=args.pgen,
                        agen=args.agen,
                        cx_algo=cx_algo,
                        mt_algo=mt_algo,
                        plot=args.plot,
                        verbose_routes=args.routes)

            now = datetime.datetime.now().strftime("%Y%m%d__%I_%M_%S%p")
            f_name = f'{cvrp.cx_algo}_{cvrp.ngen}_{cvrp.cxpb}_{cvrp.problem_set_name}__{now}'

            for i in range(1, runtime + 1):
                result = cvrp.run()

                if args.plot:
                    fig_name = f'{output}/{f_name}__RUN{i}__FIT{result["best_individual_fitness"]}.jpg'
                    result['mat_plot'].savefig(fig_name, bbox_inches='tight')
                    del result['mat_plot']

                runs["RUNS"][f"RUN_{i}"] = result
                cvrp.reset()
                print(f"\n\n============END RUN {i}============\n\n")

            print("...All runs complete")
            runs['BEST_RUN'] = min(runs['RUNS'], key=lambda run: runs['RUNS'][run]['best_individual_fitness'])
            runs['WORST_RUN'] = max(runs['RUNS'], key=lambda run: runs['RUNS'][run]['best_individual_fitness'])
            runs["AVG_FITNESS"] = sum(v['best_individual_fitness'] for v in runs['RUNS'].values()) / len(runs['RUNS'].keys())


            js_res = json.dumps(obj=runs,
                                cls=CVRPEncoder,
                                indent=args.indent)

            problem_name = runs['RUNS']['RUN_1']["problem_set_name"]
            run_data = runs['RUNS']['RUN_1']
            output_data[problem_name] = {
                "name": run_data["problem_set_name"],
                "optimal": run_data["problem_set_optimal"],
                "time": run_data["time"],
                "cost": run_data["best_individual_fitness"],
                "routes": run_data.get("routes", [])  # If routes are not available, default to empty list
            }


            print(js_res)
            
        with open(os.path.join(output, f'E_output_100_000.json'), 'w+') as fc:
            fc.write(json.dumps(output_data))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as kms:
        print("Keyboard Interrupt")
        try:
            sys.exit(1)
        except SystemExit:
            os._exit(1)
