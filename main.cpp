#include <iostream>
#include <string>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl/common/pca.h>
#include <pcl/common/transforms.h>

pcl::PointCloud<pcl::PointXYZRGB>::Ptr loadPointCloud(const std::string& filename) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    if (pcl::io::loadPLYFile<pcl::PointXYZRGB>(filename, *cloud) == -1) {
        PCL_ERROR("Couldn't read file %s \n", filename.c_str());
        return nullptr;
    }
    return cloud;
}

void findAndAlignCentralAxis(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) {
    // Calculate the centroid of the entire point cloud
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);

    // Translate the point cloud to center it along the x-axis
    Eigen::Matrix4f translation = Eigen::Matrix4f::Identity();
    translation(0, 3) = -centroid[0];
    translation(1, 3) = -centroid[1];
    translation(2, 3) = -centroid[2];
    pcl::transformPointCloud(*cloud, *cloud, translation);

    // Perform PCA to align the principal component with the x-axis
    pcl::PCA<pcl::PointXYZRGB> pca;
    pca.setInputCloud(cloud);
    Eigen::Matrix3f eigenVectors = pca.getEigenVectors();
    Eigen::Vector3f eigenValues = pca.getEigenValues();

    // Align the principal axis with the x-axis
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3, 3>(0, 0) = eigenVectors.transpose();

    // Transform the point cloud to align it
    pcl::transformPointCloud(*cloud, *cloud, transform);
}

float calculateMiddleRadius(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) {
    float radius_sum = 0.0;
    int count = 0;
    for (const auto& point : cloud->points) {
        float radius = std::sqrt(point.y * point.y + point.z * point.z);
        radius_sum += radius;
        count++;
    }
    return radius_sum / count;
}

void transformPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, float middle_radius) {
    for (auto& point : cloud->points) {
        float radius = std::sqrt(point.y * point.y + point.z * point.z);
        float theta = std::atan2(point.z, point.y);
        // Map theta to a linear coordinate (unwrap) starting from the point on the minus z-axis that is equal to the middle radius
        point.y = radius * theta;
        // Offset the z-coordinate by the middle radius
        point.z = radius - middle_radius;
    }
}

bool savePointCloud(const std::string& filename, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) {
    if (pcl::io::savePLYFile<pcl::PointXYZRGB>(filename, *cloud) == -1) {
        PCL_ERROR("Couldn't write file %s \n", filename.c_str());
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_ply_file> <aligned_ply_file> <output_ply_file>" << std::endl;
        return -1;
    }

    std::string input_filename = argv[1];
    std::string aligned_filename = argv[2];
    std::string output_filename = argv[3];

    auto cloud = loadPointCloud(input_filename);
    if (!cloud) {
        return -1;
    }

    findAndAlignCentralAxis(cloud);
    
    // Save the point cloud after aligning the central axis
    if (!savePointCloud(aligned_filename, cloud)) {
        return -1;
    }

    float middle_radius = calculateMiddleRadius(cloud);
    transformPointCloud(cloud, middle_radius);

    if (!savePointCloud(output_filename, cloud)) {
        return -1;
    }

    std::cout << "Transformation complete. Aligned output saved to " << aligned_filename << " and final output saved to " << output_filename << std::endl;
    return 0;
}
