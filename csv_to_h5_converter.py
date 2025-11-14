#!/usr/bin/env python3
"""
DRAM CSV to H5 Converter

CSV 파일들을 읽어서 깨끗한 H5 파일로 변환
입력: csv/*.csv (9개 파일)
출력: dram_iv_clean.h5 (structured array)
"""

import numpy as np
import pandas as pd
import h5py
import os
from pathlib import Path
import re


def parse_filename(filename):
    """
    파일명에서 spacer, doping 추출
    예: '15nm,1e+17.csv' -> (15.0, 1e17)
    """
    # 파일명 패턴: 15nm,1e+17.csv 또는 15nm, 1e+17.csv
    pattern = r'(\d+)nm[,\s]+(\d+\.?\d*[eE][+\-]?\d+)'
    match = re.search(pattern, filename)
    
    if match:
        spacer = float(match.group(1))
        doping = float(match.group(2))
        return spacer, doping
    else:
        raise ValueError(f"파일명 파싱 실패: {filename}")


def main():
    print("=" * 60)
    print("DRAM CSV to H5 Converter")
    print("=" * 60)
    
    # 1. CSV 파일 목록 확인
    csv_dir = Path('csv')
    csv_files = sorted(csv_dir.glob('*.csv'))
    
    print(f"\n발견된 CSV 파일: {len(csv_files)}개\n")
    for f in csv_files:
        print(f"  - {f.name}")
    
    # 파일명 파싱 테스트
    print("\n파일명 파싱 테스트:")
    test_files = ['15nm,1e+17.csv', '35nm, 1e+19.csv', '15nm_1e+19.csv']
    for tf in test_files:
        try:
            s, d = parse_filename(tf)
            print(f"  {tf:20s} -> Spacer={s}nm, Doping={d:.0e}")
        except:
            print(f"  {tf:20s} -> 파싱 실패 (건너뛰기)")
    
    # 2. CSV 데이터 읽기 및 변환
    print("\n" + "=" * 60)
    print("CSV 데이터 읽기 중...")
    print("=" * 60)
    
    # Gate length 매핑 (5개 칼럼 쌍 순서대로)
    GATE_LENGTHS = [5.0, 10.0, 20.0, 40.0, 80.0]
    
    all_records = []
    
    for csv_file in csv_files:
        try:
            # 파일명에서 파라미터 추출
            spacer, doping = parse_filename(csv_file.name)
            
            # CSV 읽기 (헤더 있음)
            df = pd.read_csv(csv_file)
            
            # 칼럼 수 확인 (10개 = 5쌍)
            if df.shape[1] != 10:
                print(f"⚠️  {csv_file.name}: 칼럼 수가 {df.shape[1]}개 (예상: 10개)")
                continue
            
            # 5개 gate length에 대해 반복
            for gate_idx, gate_length in enumerate(GATE_LENGTHS):
                # X, Y 칼럼 인덱스 (0,1 / 2,3 / 4,5 / 6,7 / 8,9)
                x_col = gate_idx * 2
                y_col = gate_idx * 2 + 1
                
                # Voltage, Current 추출
                voltages = df.iloc[:, x_col].values
                currents = df.iloc[:, y_col].values
                
                # 각 (V, I) 쌍을 레코드로 추가
                for v, i in zip(voltages, currents):
                    # 유효한 숫자인지 확인 (NaN이나 문자열 제외)
                    if pd.notna(v) and pd.notna(i):
                        try:
                            v_float = float(v)
                            i_float = float(i)
                            all_records.append((
                                spacer,
                                doping,
                                gate_length,
                                v_float,
                                i_float
                            ))
                        except (ValueError, TypeError):
                            # 변환 실패한 값 무시
                            pass
            
            print(f"✅ {csv_file.name:20s} | Spacer={spacer:2.0f}nm, Doping={doping:.0e} | 데이터 추출 완료")
            
        except Exception as e:
            print(f"❌ {csv_file.name}: 에러 - {e}")
    
    print(f"\n총 레코드 수 (중복 포함): {len(all_records):,}개")
    
    # 중복 제거
    print("\n중복 데이터 제거 중...")
    # Set을 사용하여 중복 제거 (spacer, doping, gate, voltage, current 모두 같은 경우)
    unique_records = list(set(all_records))
    removed_count = len(all_records) - len(unique_records)
    print(f"  제거된 중복 레코드: {removed_count:,}개")
    print(f"  남은 유니크 레코드: {len(unique_records):,}개")
    
    # 정렬 (spacer, doping, gate, voltage 순서로)
    unique_records.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    
    # 3. NumPy Structured Array 생성
    print("\n" + "=" * 60)
    print("NumPy Structured Array 생성 중...")
    print("=" * 60)
    
    # Structured array dtype 정의
    dt = np.dtype([
        ('spacer_length', 'f4'),  # float32
        ('doping_conc', 'f4'),
        ('gate_length', 'f4'),
        ('voltage', 'f4'),
        ('current', 'f4')
    ])
    
    # Array 생성 (중복 제거된 레코드 사용)
    data_array = np.array(unique_records, dtype=dt)
    
    print(f"\nArray 생성 완료")
    print(f"  Shape: {data_array.shape}")
    print(f"  Dtype: {data_array.dtype}")
    print(f"  Size: {data_array.nbytes / 1024:.1f} KB")
    print(f"\n처음 5개 샘플:")
    for i in range(min(5, len(data_array))):
        print(f"  {data_array[i]}")
    
    # 4. H5 파일로 저장
    print("\n" + "=" * 60)
    print("H5 파일 저장 중...")
    print("=" * 60)
    
    output_file = 'dram_iv_clean.h5'
    
    with h5py.File(output_file, 'w') as f:
        # 메인 데이터셋 생성
        dset = f.create_dataset('iv_data', data=data_array, compression='gzip', compression_opts=9)
        
        # 메타데이터 추가
        dset.attrs['description'] = 'DRAM I-V characteristics data'
        dset.attrs['columns'] = 'spacer_length, doping_conc, gate_length, voltage, current'
        dset.attrs['units'] = 'nm, cm^-3, nm, V, A'
        dset.attrs['source'] = 'Converted from CSV files'
        dset.attrs['total_records'] = len(data_array)
        
        # 데이터 범위 정보
        f.attrs['spacer_range'] = f"[{data_array['spacer_length'].min()}, {data_array['spacer_length'].max()}] nm"
        f.attrs['doping_range'] = f"[{data_array['doping_conc'].min():.0e}, {data_array['doping_conc'].max():.0e}] cm^-3"
        f.attrs['gate_range'] = f"[{data_array['gate_length'].min()}, {data_array['gate_length'].max()}] nm"
        f.attrs['voltage_range'] = f"[{data_array['voltage'].min():.2f}, {data_array['voltage'].max():.2f}] V"
        f.attrs['current_range'] = f"[{data_array['current'].min():.2e}, {data_array['current'].max():.2e}] A"
    
    print(f"\n✅ H5 파일 저장 완료: {output_file}")
    print(f"   파일 크기: {os.path.getsize(output_file) / 1024:.1f} KB")
    
    # 5. 저장된 파일 확인
    print("\n" + "=" * 60)
    print("H5 파일 검증 중...")
    print("=" * 60)
    
    with h5py.File(output_file, 'r') as f:
        # 데이터셋 정보
        dset = f['iv_data']
        print(f"\nDataset: {dset.name}")
        print(f"  Shape: {dset.shape}")
        print(f"  Dtype: {dset.dtype}")
        print(f"  Compression: {dset.compression}")
        
        # 메타데이터
        print(f"\n메타데이터:")
        for key, value in dset.attrs.items():
            print(f"  {key}: {value}")
        
        print(f"\n파일 전체 속성:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
        
        # 데이터 샘플
        data = dset[:]
        print(f"\n처음 5개 레코드:")
        for i in range(min(5, len(data))):
            print(f"  [{i}] Spacer={data[i]['spacer_length']:.0f}nm, "
                  f"Doping={data[i]['doping_conc']:.0e}, "
                  f"Gate={data[i]['gate_length']:.0f}nm, "
                  f"V={data[i]['voltage']:.2f}V, "
                  f"I={data[i]['current']:.2e}A")
    
    # 6. 사용 예시
    print("\n" + "=" * 60)
    print("사용 예시")
    print("=" * 60)
    
    with h5py.File(output_file, 'r') as f:
        data = f['iv_data'][:]
    
    # 특정 조건 필터링
    mask = (data['spacer_length'] == 15) & \
           (data['doping_conc'] == 1e17) & \
           (data['gate_length'] == 5)
    
    filtered = data[mask]
    print(f"\n1. Spacer=15nm, Doping=1e17, Gate=5nm 필터링")
    print(f"   결과: {len(filtered)}개 데이터포인트")
    if len(filtered) > 0:
        print(f"   Voltage 범위: [{filtered['voltage'].min():.2f}, {filtered['voltage'].max():.2f}] V")
    
    # 유니크 값 확인
    print(f"\n2. 파라미터 유니크 값")
    print(f"   Spacer: {np.unique(data['spacer_length'])} nm")
    print(f"   Doping: {np.unique(data['doping_conc'])} cm^-3")
    print(f"   Gate: {np.unique(data['gate_length'])} nm")
    
    # 7. 통계 요약
    print("\n" + "=" * 60)
    print("데이터 통계")
    print("=" * 60)
    
    # 파라미터별 조합 개수
    n_spacer = len(np.unique(data['spacer_length']))
    n_doping = len(np.unique(data['doping_conc']))
    n_gate = len(np.unique(data['gate_length']))
    
    print(f"\n파라미터 조합:")
    print(f"  Spacer: {n_spacer}개 값")
    print(f"  Doping: {n_doping}개 값")
    print(f"  Gate: {n_gate}개 값")
    print(f"  총 조합: {n_spacer} × {n_doping} × {n_gate} = {n_spacer * n_doping * n_gate}개")
    
    # 각 조합당 평균 데이터포인트
    avg_points = len(data) / (n_spacer * n_doping * n_gate)
    print(f"  조합당 평균 I-V 포인트: {avg_points:.1f}개")
    
    print("\n" + "=" * 60)
    print(f"✅ 변환 완료!")
    print(f"   총 {len(data):,}개 레코드가 {output_file}에 저장되었습니다.")
    print("=" * 60)


if __name__ == "__main__":
    main()

