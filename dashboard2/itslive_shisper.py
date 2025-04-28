import click
import pandas as pd
import itslive
from rich import print as rprint

# Define your function for exporting data
def export(
    itslive_catalog, input_coordinates, lat, lon, variables, outdir, format, debug
):
    """
    ITS_LIVE Global Glacier Velocity

    [i]You can try using --help at the top level and also for
    specific group subcommands.[/]
    """

    points = []
    itslive.velocity_cubes.load_catalog(itslive_catalog)
    if debug:
        rprint("Debug mode is [red]on[/]")
        rprint(f"Using: {itslive.velocity_cubes._current_catalog_url}")
    
    if input_coordinates is not None:
        # Read coordinates from the CSV file
        for index, row in input_coordinates.iterrows():
            points.append((row["lon"], row["lat"]))
    else:
        if lat and lon:
            points.append((lon, lat))

    if len(points) and format is not None:
        export_time_series(points, variables, format, outdir)
    else:
        rprint("At least one set of coordinates is needed. Use --help for assistance.")

def export_time_series(points, variables, format, outdir):
    if format == "csv":
        itslive.velocity_cubes.export_csv(points, variables, outdir)
    elif format == "netcdf":
        itslive.velocity_cubes.export_netcdf(points, variables, outdir)
    else:
        itslive.velocity_cubes.export_stdout(points, variables)

@click.command()
@click.option(
    "--itslive-catalog",
    required=False,
    default="https://its-live-data.s3.amazonaws.com/datacubes/catalog_v02.json",
    help="GeoJSON catalog with the ITS_LIVE cube metadata",
)
@click.option(
    "--input-coordinates",
    required=False,
    type=click.Path(),
    help="Input CSV file with coordinates",
)
@click.option(
    "--lat",
    type=float,
    help="Latitude, e.g. 70.1",
)
@click.option(
    "--lon",
    type=float,
    help="Longitude, e.g. -120.4",
)
@click.option(
    "--variables",
    type=click.Choice(["v", "v_error", "vy", "vx"], case_sensitive=False),
    multiple=True,
    default=["v"],
    help="Variables to export",
)
@click.option(
    "--outdir",
    type=str,
    help="Output directory",
)
@click.option(
    "--format",
    type=click.Choice(["csv", "netcdf", "stdout"]),
    help="Export format",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Verbose output",
)
def main(itslive_catalog, input_coordinates, lat, lon, variables, outdir, format, debug):
    # Read CSV file if provided
    if input_coordinates:
        input_coordinates = pd.read_csv(input_coordinates)
    
    export(
        itslive_catalog, 
        input_coordinates, 
        lat, 
        lon, 
        variables, 
        outdir, 
        format, 
        debug
    )

if __name__ == "__main__":
    main()
