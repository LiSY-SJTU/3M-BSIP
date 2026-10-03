import argparse
from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="ntM 驱动的蛋白-DNA 建模")
    parser.add_argument("--target", required=True, help="目标蛋白 PDB 路径")
    parser.add_argument("--templates", required=True, help="模板库根目录")
    parser.add_argument("--interface", required=True, help="interface.csv")
    parser.add_argument("--pred", required=True, help="pred_nt.csv")
    parser.add_argument("--outdir", required=True, help="输出目录")
    parser.add_argument("--rmsd", type=float, default=2.0, help="Kabsch RMSD 阈值")
    parser.add_argument("--clash", type=float, default=2.0, help="蛋白-核酸碰撞阈值 (Å)")
    parser.add_argument(
        "--overlap", type=float, default=0.8,
        help="DNA 糖-磷酸骨架重叠原子距离阈值 (Å；不比较碱基类型/原子)",
    )
    args = parser.parse_args()

    run_pipeline(
        target_pdb=args.target,
        templates_root=args.templates,
        interface_csv=args.interface,
        pred_nt_csv=args.pred,
        outdir=args.outdir,
        rmsd_threshold=args.rmsd,
        clash_distance=args.clash,
        overlap_threshold=args.overlap,
    )


if __name__ == "__main__":
    main()
