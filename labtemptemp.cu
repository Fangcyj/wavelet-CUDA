#include<stdio.h>
#include"/usr/include/opencv/highgui.h"
#include"cuda_runtime.h"
#include<math.h>
#include<malloc.h>

//define the texture that we use in the function later
texture<char,2,cudaReadModeElementType> teximg;
__global__ void wavelet_decom_nlevel(float* img_out , float*img_in,int nlevel,int width);
__global__ void wavelet_decom(char* img,int img_width);
/*
in the this function , img represents for the array of img,img_width is used for change 
from 2D img into 1D form.
img must be char type for there are some negative values after the wavelet decomposition
ilevel is the maximum level of waveet_transform you want to do with the image.
ilevel should not be larger than log_2(width),usually we choose 2 or 3
*/


//全局变量
__device__ float* gtemp;

int main(){
	
	//firstly load the image you want to do the wavelet transform
	IplImage* img = cvLoadImage("shepp1024.jpg",0);


	//	uchar* tmp ={0};
	cvNamedWindow("Image_show",CV_WINDOW_AUTOSIZE);
        cvShowImage("Image_show",img);
	//to compare with the image after wavelet transform,show the image before the transform

	//test the type of the image
	printf("width = %d\n height = %d\n",img->width,img->height);
	//the results shows that the img have a very different type of data
        //	printf("%d",img.at<double>(1,1));
	printf("%f\n",(img->imageData[1]+img->imageData[2])/2.0);
	//as I turn the image number from char to unchar,it can be printed well on the screen
	//to store the image which we can used to do the wavelet transform we choose a double type array
/*
	char big[img->width][img->height];
	int a = 0;
	//give the value to tmp	

	for(int i = 0; i< img->height;i++)
	{
		
		for(int j = 0; j <img->width;j++)
		{
			big[i][j] = img->imageData[i*img->width + j];
			if(big[i][j]>127) {printf("%d\n",(int)big[i][j]);a++;}
			
		}

	}	
*/		 
	printf("the size of float: %lu\n",sizeof(float));
	printf("the depth : %d\n",img->depth);
	printf("the channels : %d\n",img->nChannels);
	printf("type of store: %d\n",img->dataOrder);
	printf("origin:        %d\n",img->origin);
	printf("imageSize      %d\n",img->imageSize);
	printf("widthStep: %d\n",img->widthStep);
	//	printf("the length of data : %lu\n",sizeof(img->imageData)/sizeof(char));
	//这里我们得到的结果是图像是3通道的，所以后面的channellDesc需要修改，因此我将底下的改成了3通道。
	//开辟一段CUDA数组型内存，高度等于图像高度，宽度等于图像宽度,同时也定义了数据类型,
	//在小波变换的过程中如果想要保证数据格式不发生变化，那么就使用unsigned类型，但是这会带来一定的问题
	//在计算的过程中，小数点后面的部分就没有了。对重建会有一定的影响。
	cudaChannelFormatDesc	channelDesc = cudaCreateChannelDesc(8,0,0,0,cudaChannelFormatKindSigned);
	cudaArray	       *imgOnMemory;
	cudaMallocArray(&imgOnMemory,&channelDesc,img->height,img->width);
	//将图像扔进内存中,这里有个Step存疑
	cudaMemcpyToArray(imgOnMemory,0,0,img->imageData,img->height*img->width,cudaMemcpyHostToDevice);
	//绑定纹理引用
	cudaBindTextureToArray(&teximg,imgOnMemory,&channelDesc);

	float l = 66.6;
	printf("%c\n",(char)l);





	//定义一个新的图像img_new_texture
	IplImage* img_new_texture = cvCreateImage(cvSize(img->width,img->height),IPL_DEPTH_8S,1);
	char * temp;
	cudaMalloc((char**)&temp,img->width*img->height*sizeof(char));
	
	IplImage* img_float      = cvCreateImage(cvSize(img->width,img->height),IPL_DEPTH_32F,1);
	float *temp_float;
	cudaMalloc((float**)&temp_float,img->width*img->height*sizeof(float));
	float *odd;
	cudaMalloc((void**)&odd,img->width*img->height*sizeof(float));
	cudaMemcpyToSymbol(gtemp,&odd,sizeof(float*),size_t(0),cudaMemcpyHostToDevice);







//定义一个cuda数组用于传递img->imageData






	float  *hostimg;
	hostimg = (float *) malloc(sizeof(float)*img->height*img->width);
	for(int j = 0 ;j<img->width*img->height;j++) 
	{	hostimg[j] = (uchar)img->imageData[j]/1.0;
	//	if(hostimg[j]>0) printf("%lf\n",hostimg[j]); 
	}//printf("%lf\n",hostimg[5]);
	float  *datatran;
	cudaMalloc((float**)&datatran,img->width*img->height*sizeof(float));
	cudaMemcpy(datatran,hostimg,img->width*img->height*sizeof(float),cudaMemcpyHostToDevice);









	dim3 dimBlock(32,32);
//	dim3 dimGrid(8,8);	
	dim3 dimGrid((img->widthStep + dimBlock.x -1)/dimBlock.x,(img->height + dimBlock.y -1)/dimBlock.y);


//用纹理储存器进行计算
	wavelet_decom<<<dimGrid,dimBlock>>>(temp,img->width);
	cudaThreadSynchronize();
	//将数据传回
	cudaMemcpy(img_new_texture->imageData,temp,img->widthStep*img->height*sizeof(char),cudaMemcpyDeviceToHost);

//用普通的进行计算
	wavelet_decom_nlevel<<<dimGrid,dimBlock>>>(temp_float,datatran,1,img->width);
//	cudaDeviceSynchronize();
	cudaMemcpy(img_float->imageData,temp_float,img->width*img->height*sizeof(float),cudaMemcpyDeviceToHost);
	printf("value : %d\n",img_float->imageData[1]);




//显示图片
//	for(int flag = 0;flag<img->height*img->width;flag++) img_new_texture->imageData[flag] = img->imageData[flag]-128;
	cvNamedWindow("wavelet",CV_WINDOW_AUTOSIZE);
        cvShowImage("wavelet",img_new_texture);
	
	cvNamedWindow("odinary-way",CV_WINDOW_AUTOSIZE);
	cvShowImage("odinary-way",img_float);
	
//      保存图片到地
	 cvSaveImage("./wavelet.jpg",img_float);
	 cvSaveImage("./texture.jpg",img_new_texture);



	cudaUnbindTexture(&teximg);
	cudaFreeArray(imgOnMemory);
	cudaFree(temp);
	cudaFree(temp_float);
	cudaFree(datatran);
	cudaFree(odd);
	free(hostimg);
	cvWaitKey(0);
     	cvReleaseImage(&img);
	cvReleaseImage(&img_new_texture);
	cvReleaseImage(&img_float);
    	cvDestroyWindow("Image_show");
	cvDestroyWindow("wavelet");
     	return 0;

	
}



//这个函数用于做图像小波分解,使用纹理内存，后来意识到纹理内存只能读取，所以这个函数只能实现一级小波分解
//基本上没啥用
__global__ void wavelet_decom(char* img,int img_width)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;		
/*
	if(x <img_width/2)
	img[y*img_width + x] = 15;
	else 
	img[y*img_width + x] = -100;
*/
	if(x<img_width/2&&y<img_width/2){
		img[y*img_width + x]                             = ((tex2D(teximg,2*x+1,2*y+1)+ tex2D(teximg,2*x,2*y+1))+tex2D(teximg,2*x+1,2*y)+tex2D(teximg,2*x,2*y))/4 - 127;
		img[y*img_width + x + img_width/2]               = ((tex2D(teximg,2*x+1,2*y+1)- tex2D(teximg,2*x,2*y+1))+tex2D(teximg,2*x+1,2*y)-tex2D(teximg,2*x,2*y))/2 ;
	
	//同样对y方向也做一样的处理
                img[(y+img_width/2)*img_width + x]               = ((tex2D(teximg,2*x+1,2*y+1)+ tex2D(teximg,2*x,2*y+1))-tex2D(teximg,2*x+1,2*y)-tex2D(teximg,2*x,2*y))/2 ;
                img[(y+img_width/2)*img_width + x + img_width/2] = ((tex2D(teximg,2*x+1,2*y+1)- tex2D(teximg,2*x,2*y+1))-tex2D(teximg,2*x+1,2*y)+tex2D(teximg,2*x,2*y))   ;
        }
	if(img[y*img_width+x]<5&&img[y*img_width+x]>-5)	img[y*img_width + x] = 0;

}
//这个是不用texture的,可以做多层
__global__ void wavelet_decom_nlevel(float* img_out , float*img_in,int nlevel,int width){
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	int widthtemp = width;
	gtemp[y*width + x] =  (uchar) img_in[y*width+x]/256.0;
	__syncthreads();

//	img_out[y*width + x] =  img_out[y*width +x ] +0.5;
//	for(int i = 0; i<nlevel ; i++){
		widthtemp = widthtemp/2;
		//先对x做
		if(x<widthtemp&&y<widthtemp){
			img_out[y*width+x] =1/256.0*((img_in[2*y*width+2*x] +  img_in[2*y*width+2*x+1] + img_in[(2*y+1)*width+2*x] + img_in[(2*y+1)*width+2*x+1])/4.0);
	//		__syncthreads();
			img_out[y*width+x+widthtemp] =1/256.0* (-img_in[2*y*width+2*x] +  img_in[2*y*width+2*x+1] - img_in[(2*y+1)*width+2*x] + img_in[(2*y+1)*width+2*x+1])/2.0;
	//		__syncthreads();
			img_out[(y+widthtemp)*width+x] = 1/256.0*(-img_in[2*y*width+2*x] -  img_in[2*y*width+2*x+1] + img_in[(2*y+1)*width+2*x] + img_in[(2*y+1)*width+2*x+1])/2.0;
	//		__syncthreads();
			img_out[(y+widthtemp)*width+x+widthtemp] =1/256.0* (img_in[2*y*width+2*x] -  img_in[2*y*width+2*x+1] - img_in[(2*y+1)*width+2*x] + img_in[(2*y+1)*width+2*x+1])/1.0;
	//		__syncthreads();

		}
	

/*
		
	if(x<widthtemp&&y<widthtemp){
		temp[y*width+x] = img_out[2*y*width+2*x];
		__syncthreads();
		temp[y*width+x+widthtemp] = img_out[2*y*width+2*x+1];
		__syncthreads();
		temp[(y+widthtemp)*width+x] = img_out[(2*y+1)*width+2*x];
		__syncthreads();
		temp[(y+widthtemp)*width+x+widthtemp] = img_out[(2*y+1)*width+2*x+1];
		__syncthreads();
	}	
*/
//用于测试 全局变量是否有效
//        img_out[y*width+x]=gtemp[y*width+x];
	__syncthreads();	
	}







































