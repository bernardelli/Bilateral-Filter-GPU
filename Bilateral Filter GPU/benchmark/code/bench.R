library(ggplot2) 

file_list <- Sys.glob("*kom.txt")

file = file_list[2]

gpu_name = readLines(file, n=1) 

print(gpu_name)

data <- read.table(file, header = TRUE, sep = "\t", skip = 1)

data_subset <- subset( data, scale_eps == 11 & scale_xy == 11 & kernel_eps_size == 11 & kernel_xy_size > 1)

data_subset$kernel_xy_size <- factor(data_subset$kernel_xy_size)
#TRY MEDIAN INSTEAD OF MEAN!
#filling
ggplot(data_subset, aes(x=Image.size, y=cubefilling, color=kernel_xy_size)) +
geom_point(alpha = 0.4) + 
stat_summary(fun.y = median, geom="line") + 
scale_colour_brewer(name="Spatial kernel length",palette = "Set1") +
ggtitle(paste("Filling on", gpu_name)) +# for the main title
xlab("Image size") + # for the x axis label
ylab("Runtime (ms)") + # for the y axis label
ggsave("out0.png",width = 7.5, height = 5)



#conv
ggplot(data_subset, aes(x=Image.size, y=convolution, color=kernel_xy_size)) +
geom_point(alpha = 0.4) + 
stat_summary(fun.y = median, geom="line") + 
scale_colour_brewer(name="Spatial kernel length",palette = "Set1") +
ggtitle(paste("Convolution on", gpu_name)) +# for the main title
xlab("Image size") + # for the x axis label
ylab("Runtime (ms)") + # for the y axis label


ggsave("out1.png",width = 7.5, height = 5)

#slicing
ggplot(data_subset, aes(x=Image.size, y=slicing, color=kernel_xy_size)) +
geom_point(alpha = 0.4) + 
stat_summary(fun.y = median, geom="line") + 
scale_colour_brewer(name="Spatial kernel length",palette = "Set1") +
ggtitle(paste("Slicing on", gpu_name)) +# for the main title
xlab("Image size") + # for the x axis label
ylab("Runtime (ms)") + # for the y axis label


ggsave("out2.png",width = 7.5, height = 5)

#allocate
ggplot(data_subset, aes(x=Image.size, y=time.to.allocate, color=kernel_xy_size)) +
geom_point(alpha = 0.4) + 
stat_summary(fun.y = median, geom="line") + 
scale_colour_brewer(name="Spatial kernel length",palette = "Set1") +
ggtitle(paste("Allocation on", gpu_name)) +# for the main title
xlab("Image size") + # for the x axis label
ylab("Runtime (ms)") + # for the y axis label
coord_cartesian(ylim = quantile(data_subset$time.to.allocate, c(0.1, 0.9)))

ggsave("out3.png",width = 7.5, height = 5)

#copy


ggplot(data_subset, aes(x=Image.size, y=time.copy.memory, color=kernel_xy_size)) +
geom_point(alpha = 0.4) + 
stat_summary(fun.y = median, geom="line") + 
scale_colour_brewer(name="Spatial kernel length",palette = "Set1") +
ggtitle(paste("Memory transfer on", gpu_name)) +# for the main title
xlab("Image size") + # for the x axis label
ylab("Runtime (ms)") + # for the y axis label
coord_cartesian(ylim = quantile(data_subset$time.copy.memory, c(0.1, 0.9)))

ggsave("out4.png",width = 7.5, height = 5)

#free

ggplot(data_subset, aes(x=Image.size, y=time.to.free, color=kernel_xy_size)) +
geom_point(alpha = 0.4) + 
stat_summary(fun.y = median, geom="line") + 
scale_colour_brewer(name="Spatial kernel length",palette = "Set1") +
ggtitle(paste("Time to free on", gpu_name)) +# for the main title
xlab("Image size") + # for the x axis label
ylab("Runtime (ms)") + # for the y axis label
coord_cartesian(ylim = quantile(data_subset$time.to.free, c(0.1, 0.9)))
ggsave("out5.png",width = 7.5, height = 5)
#complete

ggplot(data_subset, aes(x=Image.size, y=komplete.time, color=kernel_xy_size)) +
geom_point(alpha = 0.4) + 
stat_summary(fun.y = median, geom="line") + 
scale_colour_brewer(name="Spatial kernel length",palette = "Set1") +
ggtitle(paste("Full run on", gpu_name)) +# for the main title
xlab("Image size") + # for the x axis label
ylab("Runtime (ms)") + # for the y axis label
coord_cartesian(ylim = quantile(data_subset$komplete.time, c(0.1, 0.9)))
ggsave("out6.png",width = 7.5, height = 5)

#filling + convolution + slicing 
#data_subset["fcs.time"] <- 0
data_subset$fcs_time <- data_subset$cubefilling + data_subset$convolution + data_subset$slicing



ggplot(data_subset, aes(x=Image.size, y=fcs_time, color=kernel_xy_size)) +
geom_point(alpha = 0.4) + 
stat_summary(fun.y = median, geom="line") + 
scale_colour_brewer(name="Spatial kernel length",palette = "Set1") +
ggtitle(paste("Filling + Convolution + Slicing on", gpu_name)) +# for the main title
xlab("Image size") + # for the x axis label
ylab("Runtime (ms)") + # for the y axis label
coord_cartesian(ylim = quantile(data_subset$fcs_time, c(0.001, 0.999)))
ggsave("out7.png",width = 7.5, height = 5)