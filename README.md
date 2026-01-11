# Long-Tail Post-Processing (LTPP) for Temporal Knowledge Graph Reasoning 

## Environment Setup 

#### 1. Create the Conda Environment

```bash
conda create -n ltpp python=3.10.18
```

#### 2. Activate the Environment

```bash
conda activate ltpp
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Dataset Preparation 

#### 1. Unzip Data 

```bash
unzip data.zip
```

#### 2. Dataset Placement 

```bash
cp -r data/ ./backbone1_convtranse/
cp -r data/ ./backbone2_renet/
cp -r data/ ./backbone3_regcn/
cp -r data/ ./backbone4_tirgn/
```

## Implementations 

Please check each Bourne shell script for detailed implementation commands. 
