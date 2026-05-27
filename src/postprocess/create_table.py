import argparse

import pandas as pd


def convert_to_string_table(filename: str):
    def format_clap(row: pd.Series, idx: int):
        return f"{row['clap_'+str(idx)+'_mean']:.03f}" + " ± " + f"{row['clap_'+str(idx)+'_std']:.02f}"

    df = pd.read_csv(filename)

    df["CLAP(clean)"] = df.apply(lambda row: format_clap(row, 1), axis=1)
    df["CLAP(corrupt)"] = df.apply(lambda row: format_clap(row, 2), axis=1)
    df_output = df[
        [
            "Block",
            "frechet_distance",
            "frechet_audio_distance",
            "kullback_leibler_divergence_sigmoid",
            "lsd",
            "psnr",
            "ssim",
            "CLAP(clean)",
            "CLAP(corrupt)",
        ]
    ]
    df_output.columns = ["Clean layers", "FD", "FAD", "KL", "LSD", "PSNR", "SSIM", "CLAP(clean)", "CLAP(corrupt)"]

    table = "+-----------------------+-----------------------------------------------------------------------------------------------------------------------+----------------------------------------------+\n"
    table += "|                       |                                             Music Alignment (to no_layers)                                            |                Text Alignment                |\n"
    table += "|     Clean layers      +-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+----------------------+-----------------------+\n"
    table += "|                       |        FD         |        FAD        |        KL         |        LSD        |       PSNR        |       SSIM        |     CLAP(clean)      |    CLAP(corrupt)      |\n"
    table += "+-----------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+----------------------+-----------------------+\n"

    for _, row in df_output.iterrows():
        table += f"| {row['Clean layers']:<21} | {abs(row['FD']):>17.03f} | {abs(row['FAD']):>17.03f} | {abs(row['KL']):>17.03f} | {abs(row['LSD']):>17.03f} | "
        if pd.notna(row["PSNR"]):
            table += f"{abs(row['PSNR']):>17.03f} | "
        else:
            table += f"{'-':>17} | "
        if pd.notna(row["SSIM"]):
            table += f"{abs(row['SSIM']):>17.03f} | "
        else:
            table += f"{'-':>17} | "
        table += f"{row['CLAP(clean)']:<20} | {row['CLAP(corrupt)']:<21} |\n"

    table += "+-----------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+----------------------+-----------------------+"

    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=True)
    args = parser.parse_args()
    print(convert_to_string_table(args.filename))
