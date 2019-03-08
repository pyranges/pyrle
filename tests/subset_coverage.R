## library(S4Vectors)
## library(GenomicRanges)
## library(rtracklayer)
## f = "/home/endrebak/code/pyranges/pyranges/example_data/chipseq.bed"
## gr = import(f)

#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

f1 = args[1]
start = strtoi(args[2])
end = strtoi(args[3])
rf = args[4]

print(args)
print(f1)
print(start)
print(end)
print(rf)

print("We are starting in R! We are starting in R! We are starting in R! We are starting in R! We are starting in R! We are starting in R! We are starting in R! ")


## library(S4Vectors)
suppressMessages(library(S4Vectors))


df1 = read.table(f1, sep="\t", header=TRUE)

print("read table 1")

sum1 = sum(df1$Runs)

print("found sum 1")

df1$Values = df1$Values * 1.0

print(sum1)

print(df1)

r1 = Rle(df1$Values, df1$Runs)

print(r1)

if (start > sum1){
  start = sum1
}
if (end > sum1){
  end = sum1
}

print("start end")
print(start)
print(end)
result = r1[start:end]

print(result)

df = data.frame(Runs=runLength(result), Values=runValue(result))

write.table(df, rf, sep="\t")
