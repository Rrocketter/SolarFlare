# #!/usr/bin/env python
# """
# Solar Data Download Script
#
# This script downloads and preprocesses solar data from various sources
# for space weather forecasting.
# """
#
# import argparse
# from pathlib import Path
# import sys
# import os
# import datetime
# from solar_data_download import (
#     download_hmi_data,
#     download_aia_data,
#     download_goes_data,
#     extract_flare_events,
#     preprocess_hmi_data,
#     create_visualizations
# )
# from data_management import SolarDataManager
#
#
# def parse_args():
#     """Parse command line arguments"""
#     parser = argparse.ArgumentParser(
#         description="Download and preprocess solar data for space weather forecasting"
#     )
#
#     # Existing arguments
#     parser.add_argument(
#         "--start-date",
#         type=str,
#         default="2018-01-01",
#         help="Start date in YYYY-MM-DD format"
#     )
#
#     parser.add_argument(
#         "--end-date",
#         type=str,
#         default="2018-01-31",
#         help="End date in YYYY-MM-DD format"
#     )
#
#     parser.add_argument(
#         "--data-types",
#         type=str,
#         default="hmi,aia,goes,soho,noaa_ar", # Updated to include new data sources
#         help="Comma-separated list of data types to download"
#     )
#
#     parser.add_argument(
#         "--wavelengths",
#         type=str,
#         default="171,193,304",
#         help="Comma-separated list of AIA wavelengths to download"
#     )
#
#     parser.add_argument(
#         "--soho-instruments",
#         type=str,
#         default="c2,c3",
#         help="Comma-separated list of SOHO instruments (e.g., c2,c3 for LASCO C2 and C3)"
#     )
#
#     # Rest of existing arguments
#     parser.add_argument(
#         "--cadence",
#         type=str,
#         default="12h",
#         help="Data cadence (e.g., '12h', '1d')"
#     )
#
#     parser.add_argument(
#         "--max-files",
#         type=int,
#         default=10,
#         help="Maximum number of files to download per data type"
#     )
#
#     parser.add_argument(
#         "--max-storage",
#         type=float,
#         default=90,
#         help="Maximum storage to use in GB"
#     )
#
#     parser.add_argument(
#         "--incremental",
#         action="store_true",
#         help="Download data incrementally to manage storage"
#     )
#
#     parser.add_argument(
#         "--step-days",
#         type=int,
#         default=30,
#         help="Number of days to download in each incremental step"
#     )
#
#     parser.add_argument(
#         "--visualize",
#         action="store_true",
#         help="Create visualizations of the downloaded data"
#     )
#
#     parser.add_argument(
#         "--output-dir",
#         type=str,
#         default="./solar_data",
#         help="Base directory for data storage"
#     )
#
#     return parser.parse_args()
#
#
# # def main():
# #     # Parse arguments
# #     args = parse_args()
# #
# #     # Create data manager
# #     data_manager = SolarDataManager(base_dir=args.output_dir, max_storage_gb=args.max_storage)
# #
# #     # Parse data types
# #     data_types = [dt.strip() for dt in args.data_types.split(',')]
# #
# #     # Parse wavelengths
# #     if args.wavelengths:
# #         wavelengths = [int(wl.strip()) for wl in args.wavelengths.split(',')]
# #     else:
# #         wavelengths = [171, 193, 304]  # Default wavelengths
# #
# #     print(f"Starting solar data download process...")
# #     print(f"Date range: {args.start_date} to {args.end_date}")
# #     print(f"Data types: {data_types}")
# #     print(f"Storage limit: {args.max_storage} GB")
# #
# #     # Estimate total download size
# #     est_size = data_manager.estimate_download_size(
# #         (args.start_date, args.end_date),
# #         data_types,
# #         args.cadence
# #     )
# #     print(f"Estimated total download size: {est_size:.2f} GB")
# #
# #     # Check if we have enough space
# #     storage_report = data_manager.get_storage_report()
# #     print(f"Current storage usage: {storage_report['total_gb']:.2f} GB")
# #     print(f"Available storage: {storage_report['available_gb']:.2f} GB")
# #
# #     # Download data
# #     hmi_files = []
# #     aia_files = []
# #     goes_files = []
# #     soho_files = []  # Add this for SOHO files
# #     noaa_ar_files = []  # Add this for NOAA AR files
# #
# #     if 'hmi' in data_types:
# #         if args.incremental:
# #             print(f"Downloading HMI data incrementally...")
# #             hmi_files = data_manager.download_incremental_data(
# #                 download_hmi_data,
# #                 args.start_date,
# #                 args.end_date,
# #                 step_days=args.step_days,
# #                 sample_cadence=args.cadence,
# #                 max_files=args.max_files,
# #                 data_type='hmi',
# #                 dirs=data_manager.dirs
# #             )
# #         else:
# #             print(f"Downloading HMI data...")
# #             hmi_files = download_hmi_data(
# #                 args.start_date,
# #                 args.end_date,
# #                 dirs=data_manager.dirs,
# #                 sample_cadence=args.cadence,
# #                 max_files=args.max_files
# #             )
# #             data_manager.register_files(hmi_files, 'hmi')
# #
# #     if 'aia' in data_types:
# #         if args.incremental:
# #             print(f"Downloading AIA data incrementally...")
# #             for wl in wavelengths:
# #                 wl_files = data_manager.download_incremental_data(
# #                     download_aia_data,
# #                     args.start_date,
# #                     args.end_date,
# #                     step_days=args.step_days,
# #                     wavelengths=[wl],
# #                     sample_cadence=args.cadence,
# #                     max_files=args.max_files,
# #                     data_type=f'aia_{wl}',
# #                     dirs=data_manager.dirs
# #                 )
# #                 aia_files.extend(wl_files)
# #         else:
# #             print(f"Downloading AIA data for wavelengths {wavelengths}...")
# #             aia_files = download_aia_data(
# #                 args.start_date,
# #                 args.end_date,
# #                 wavelengths=wavelengths,
# #                 sample_cadence=args.cadence,
# #                 max_files=args.max_files,
# #                 dirs=data_manager.dirs
# #             )
# #             data_manager.register_files(aia_files, 'aia')
# #
# #     if 'goes' in data_types:
# #         print(f"Downloading GOES X-ray flux data...")
# #         goes_files = download_goes_data(args.start_date, args.end_date, dirs=data_manager.dirs)
# #         data_manager.register_files(goes_files, 'goes')
# #
# #     # Add SOHO data download
# #     if 'soho' in data_types:
# #         # Parse SOHO instruments if specified
# #         soho_instruments = args.soho_instruments.split(',') if hasattr(args,
# #                                                                        'soho_instruments') and args.soho_instruments else None
# #
# #         if args.incremental:
# #             print(f"Downloading SOHO data incrementally...")
# #             soho_files = data_manager.download_incremental_data(
# #                 download_soho_data,
# #                 args.start_date,
# #                 args.end_date,
# #                 step_days=args.step_days,
# #                 instruments=soho_instruments,
# #                 sample_cadence=args.cadence,
# #                 max_files=args.max_files,
# #                 data_type='soho',
# #                 dirs=data_manager.dirs
# #             )
# #         else:
# #             print(f"Downloading SOHO data...")
# #             soho_files = download_soho_data(
# #                 args.start_date,
# #                 args.end_date,
# #                 dirs=data_manager.dirs,
# #                 instruments=soho_instruments,
# #                 sample_cadence=args.cadence,
# #                 max_files=args.max_files
# #             )
# #             data_manager.register_files(soho_files, 'soho')
# #
# #     # Add NOAA Active Region data download
# #     if 'noaa_ar' in data_types:
# #         if args.incremental:
# #             print(f"Downloading NOAA AR data incrementally...")
# #             noaa_ar_files = data_manager.download_incremental_data(
# #                 download_noaa_ar_data,
# #                 args.start_date,
# #                 args.end_date,
# #                 step_days=args.step_days,
# #                 data_type='noaa_ar',
# #                 dirs=data_manager.dirs
# #             )
# #         else:
# #             print(f"Downloading NOAA Active Region data...")
# #             noaa_ar_files = download_noaa_ar_data(
# #                 args.start_date,
# #                 args.end_date,
# #                 dirs=data_manager.dirs
# #             )
# #             data_manager.register_files(noaa_ar_files, 'noaa_ar')
# #
# #     # Process GOES data to extract flare events
# #     if goes_files:
# #         print("Extracting flare events from GOES data...")
# #         flares_df = extract_flare_events(goes_files)
# #         if not flares_df.empty:
# #             print(f"Extracted {len(flares_df)} flare events")
# #             flares_df.to_csv(Path(args.output_dir) / "processed" / "flare_events.csv", index=False)
# #         else:
# #             print("No flare events found in the data")
# #
# #     # Update visualizations to include new data sources
# #     if args.visualize:
# #         print("Creating visualizations...")
# #         create_visualizations(
# #             hmi_files,
# #             aia_files,
# #             goes_files,
# #             soho_files,  # Add SOHO files
# #             noaa_ar_files,  # Add NOAA AR files
# #             Path(args.output_dir) / "visualizations"
# #         )
# #
# #     # Generate storage report
# #     data_manager.visualize_storage(Path(args.output_dir) / "visualizations" / "storage_report.png")
# #     data_manager.visualize_date_coverage(Path(args.output_dir) / "visualizations" / "data_coverage.png")
# #
# #     print("\nData download and preprocessing complete!")
# #     print(f"Processed data saved to: {Path(args.output_dir) / 'processed'}")
# #     print(f"Visualizations saved to: {Path(args.output_dir) / 'visualizations'}")
# #
# #     # Print final storage usage
# #     final_report = data_manager.get_storage_report()
# #     print(f"Final storage usage: {final_report['total_gb']:.2f} GB ({final_report['used_percent']:.1f}%)")
# #     print(f"Available storage: {final_report['available_gb']:.2f} GB")
# #
# # if __name__ == "__main__":
# #     main()
#
# def main():
#     # Parse arguments
#     args = parse_args()
#
#     # Create data manager with physics-based priorities
#     data_manager = SolarDataManager(
#         base_dir=args.output_dir,
#         max_storage_gb=args.max_storage
#     )
#
#     # Parse data types and set priorities
#     data_types = [dt.strip() for dt in args.data_types.split(',')]
#     priorities = {
#         'hmi': 9,  # Highest priority - magnetic field data
#         'goes': 8,  # Flare timing/classification
#         'noaa_ar': 7,  # Active region metadata
#         'aia': 6,  # Multi-wavelength context
#         'soho': 5  # Lowest priority - CME confirmation
#     }
#
#     # Download and process each data type
#     datasets = {}
#     for data_type in data_types:
#         print(f"\nProcessing {data_type.upper()} data...")
#
#         # Get the appropriate download function
#         if data_type == 'hmi':
#             download_func = download_hmi_data
#         elif data_type == 'aia':
#             download_func = download_aia_data
#         elif data_type == 'goes':
#             download_func = download_goes_data
#         elif data_type == 'noaa_ar':
#             download_func = download_noaa_ar_data
#         elif data_type == 'soho':
#             download_func = download_soho_data
#         else:
#             print(f"Unknown data type: {data_type}")
#             continue
#
#         # Download data
#         if args.incremental:
#             print(f"Downloading {data_type} data incrementally...")
#             files = data_manager.download_incremental_data(
#                 download_func,
#                 args.start_date,
#                 args.end_date,
#                 step_days=args.step_days,
#                 data_type=data_type,
#                 dirs=data_manager.dirs,
#                 priority=priorities[data_type],
#                 **({'wavelengths': args.wavelengths} if data_type == 'aia' else {}),
#                 **({'instruments': args.soho_instruments} if data_type == 'soho' else {})
#             )
#         else:
#             print(f"Downloading {data_type} data...")
#             files = download_func(
#                 args.start_date,
#                 args.end_date,
#                 dirs=data_manager.dirs,
#                 **({'wavelengths': args.wavelengths} if data_type == 'aia' else {}),
#                 **({'instruments': args.soho_instruments} if data_type == 'soho' else {})
#             )
#             data_manager.register_files(files, data_type, priority=priorities[data_type])
#
#         datasets[data_type] = files
#
#     # Process and visualize data
#     if 'goes' in datasets:
#         print("\nProcessing GOES flare data...")
#         flares_df = extract_flare_events(datasets['goes'])
#         if not flares_df.empty:
#             print(f"Extracted {len(flares_df)} flare events")
#             flares_df.to_csv(data_manager.dirs['processed'] / "flare_events.csv", index=False)
#
#             # Add flare dates to data manager
#             data_manager.flare_dates = pd.to_datetime(flares_df['time'].dt.date.unique())
#
#     if 'hmi' in datasets:
#         print("\nProcessing HMI data...")
#         preprocess_hmi_data(datasets['hmi'], data_manager.dirs['processed'])
#
#     if args.visualize:
#         print("\nCreating visualizations...")
#         create_visualizations(
#             datasets.get('hmi', []),
#             datasets.get('aia', []),
#             datasets.get('goes', []),
#             datasets.get('soho', []),
#             datasets.get('noaa_ar', []),
#             data_manager.dirs['visualizations']
#         )
#
#     # Generate final reports
#     print("\nGenerating storage report...")
#     data_manager.visualize_storage(data_manager.dirs['visualizations'] / "storage_report.png")
#     data_manager.visualize_date_coverage(data_manager.dirs['visualizations'] / "data_coverage.png")
#
#     # Print final statistics
#     final_report = data_manager.get_storage_report()
#     print(f"\nFinal storage usage: {final_report['total_gb']:.2f} GB")
#     print(f"Available storage: {final_report['available_gb']:.2f} GB")
#     print(f"Data distribution by type:")
#     for dtype, size in final_report['by_type_gb'].items():
#         print(f"  {dtype}: {size:.2f} GB ({final_report['by_type_percent'][dtype]:.1f}%)")
#
#
# if __name__ == "__main__":
#     main()


import argparse
from pathlib import Path
import sys
import os
import datetime
import pandas as pd
from solar_data_download import (
    download_hmi_data,
    download_aia_data,
    download_goes_data,
    download_soho_data,
    extract_flare_events,
    preprocess_hmi_data,
    create_visualizations
)
from data_management import SolarDataManager


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Download and preprocess solar data for space weather forecasting"
    )

    # Existing arguments
    parser.add_argument(
        "--start-date",
        type=str,
        default="2018-01-01",
        help="Start date in YYYY-MM-DD format"
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default="2018-01-31",
        help="End date in YYYY-MM-DD format"
    )

    parser.add_argument(
        "--data-types",
        type=str,
        default="hmi,aia,goes,soho",  # Updated to include new data sources
        help="Comma-separated list of data types to download"
    )

    parser.add_argument(
        "--wavelengths",
        type=str,
        default="171,193,304",
        help="Comma-separated list of AIA wavelengths to download"
    )

    parser.add_argument(
        "--soho-instruments",
        type=str,
        default="c2,c3",
        help="Comma-separated list of SOHO instruments (e.g., c2,c3 for LASCO C2 and C3)"
    )

    # Rest of existing arguments
    parser.add_argument(
        "--cadence",
        type=str,
        default="12h",
        help="Data cadence (e.g., '12h', '1d')"
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=10,
        help="Maximum number of files to download per data type"
    )

    parser.add_argument(
        "--max-storage",
        type=float,
        default=90,
        help="Maximum storage to use in GB"
    )

    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Download data incrementally to manage storage"
    )

    parser.add_argument(
        "--step-days",
        type=int,
        default=30,
        help="Number of days to download in each incremental step"
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualizations of the downloaded data"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./solar_data",
        help="Base directory for data storage"
    )

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Create data manager with physics-based priorities
    data_manager = SolarDataManager(
        base_dir=args.output_dir,
        max_storage_gb=args.max_storage
    )

    # Parse data types and set priorities
    data_types = [dt.strip() for dt in args.data_types.split(',')]
    priorities = {
        'hmi': 9,  # Highest priority - magnetic field data
        'goes': 8,  # Flare timing/classification
        'aia': 7,  # Multi-wavelength context
        'soho': 5  # Lowest priority - CME confirmation
    }

    # Process wavelengths for AIA
    wavelengths = [int(wl.strip()) for wl in args.wavelengths.split(',')] if args.wavelengths else [171, 193, 304]

    # Process SOHO instruments
    soho_instruments = [inst.strip() for inst in args.soho_instruments.split(',')] if args.soho_instruments else ['c2',
                                                                                                                  'c3']

    datasets = {}
    for data_type in data_types:
        print(f"\nProcessing {data_type.upper()} data...")

        if data_type == 'hmi':
            download_func = download_hmi_data
            extra_args = {'sample_cadence': args.cadence, 'max_files': args.max_files}
        elif data_type == 'aia':
            download_func = download_aia_data
            extra_args = {'wavelengths': wavelengths, 'sample_cadence': args.cadence, 'max_files': args.max_files}
        elif data_type == 'goes':
            download_func = download_goes_data
            extra_args = {}
        elif data_type == 'soho':
            download_func = download_soho_data
            extra_args = {'instruments': soho_instruments, 'sample_cadence': args.cadence, 'max_files': args.max_files}
        else:
            print(f"Unknown data type: {data_type}")
            continue

        if args.incremental:
            print(f"Downloading {data_type} data incrementally...")
            files = data_manager.download_incremental_data(
                download_func,
                args.start_date,
                args.end_date,
                step_days=args.step_days,
                data_type=data_type,
                dirs=data_manager.dirs,
                priority=priorities[data_type],
                **extra_args
            )
        else:
            print(f"Downloading {data_type} data...")
            files = download_func(
                args.start_date,
                args.end_date,
                dirs=data_manager.dirs,
                **extra_args
            )
            data_manager.register_files(files, data_type, priority=priorities[data_type])

        datasets[data_type] = files

    if 'goes' in datasets:
        print("\nProcessing GOES flare data...")
        flares_df = extract_flare_events(datasets['goes'])
        if not flares_df.empty:
            print(f"Extracted {len(flares_df)} flare events")
            flares_df.to_csv(data_manager.dirs['processed'] / "flare_events.csv", index=False)

            # Add flare dates to data manager
            data_manager.flare_dates = pd.to_datetime(flares_df['time'].dt.date.unique())

    if 'hmi' in datasets:
        print("\nProcessing HMI data...")
        preprocess_hmi_data(datasets['hmi'], data_manager.dirs['processed'])

    if args.visualize:
        print("\nCreating visualizations...")
        create_visualizations(
            datasets.get('hmi', []),
            datasets.get('aia', []),
            datasets.get('goes', []),
            datasets.get('soho', []),
            data_manager.dirs['visualizations']
        )

    print("\nGenerating storage report...")
    data_manager.visualize_storage(data_manager.dirs['visualizations'] / "storage_report.png")
    data_manager.visualize_date_coverage(data_manager.dirs['visualizations'] / "data_coverage.png")

    final_report = data_manager.get_storage_report()
    print(f"\nFinal storage usage: {final_report['total_gb']:.2f} GB")
    print(f"Available storage: {final_report['available_gb']:.2f} GB")
    print(f"Data distribution by type:")
    for dtype, size in final_report['by_type_gb'].items():
        print(f"  {dtype}: {size:.2f} GB ({final_report['by_type_percent'][dtype]:.1f}%)")


if __name__ == "__main__":
    main()