from tgfb_smad.defaults import default_initial_conditions, default_parameters
from tgfb_smad.simulate import simulate_timecourse


def main() -> None:
    p = default_parameters()
    y0 = default_initial_conditions(p)
    res = simulate_timecourse(ligand=1.0, mapk=0.2, pi3k=0.2, params=p, y0=y0, t_end=240, n_points=500)
    print(res.df.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
