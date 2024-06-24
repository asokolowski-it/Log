#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <limits>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Min_circle_2.h>
#include <CGAL/Min_circle_2_traits_2.h>

// Define the CGAL types
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Min_circle_2_traits_2<K> Traits;
typedef CGAL::Min_circle_2<Traits> Min_circle;

struct Circle {
    Eigen::Vector2f center;
    float radius;
};

Circle findMinEnclosingCircle(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& segment) {
    std::vector<K::Point_2> points;
    for (const auto& point : segment->points) {
        points.push_back(K::Point_2(point.y, point.z));
    }

    try {
        Min_circle mc(points.begin(), points.end(), true);
        K::Point_2 center = mc.circle().center();
        float radius = std::sqrt(mc.circle().squared_radius());

        return {Eigen::Vector2f(center.x(), center.y()), radius};
    } catch (const CGAL::Assertion_exception& e) {
        std::cerr << "CGAL assertion exception: " << e.what() << std::endl;
        return {Eigen::Vector2f(0, 0), 0};  // Return a default circle in case of error
    }
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr loadPointCloud(const std::string& filename) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    if (pcl::io::loadPLYFile<pcl::PointXYZRGB>(filename, *cloud) == -1) {
        PCL_ERROR("Couldn't read file %s \n", filename.c_str());
        return nullptr;
    }
    return cloud;
}

void transformPointCloudIncrementally(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, float segment_length = 1.0f) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    float x_min = std::numeric_limits<float>::max();
    float x_max = std::numeric_limits<float>::lowest();

    // Determine the range of x values
    for (const auto& point : cloud->points) {
        if (point.x < x_min) x_min = point.x;
        if (point.x > x_max) x_max = point.x;
    }

    // Segment the point cloud and transform each segment
    for (float x = x_min; x <= x_max; x += segment_length) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr segment(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (const auto& point : cloud->points) {
            if (point.x >= x && point.x < x + segment_length) {
                segment->points.push_back(point);
            }
        }

        if (!segment->points.empty()) {
            // Compute the minimum enclosing circle for the segment
            Circle min_circle = findMinEnclosingCircle(segment);

            // Transform each point in the segment
            for (auto& point : segment->points) {
                float y = point.y - min_circle.center[0];
                float z = point.z - min_circle.center[1];
                float radius = std::sqrt(y * y + z * z);
                float theta = std::atan2(y, z);

                if (theta < 0) {
                    theta += 2 * M_PI;
                }

                point.y = theta * min_circle.radius;
                point.z = radius - min_circle.radius;
                transformed_cloud->points.push_back(point);
            }
        }
    }

    cloud = transformed_cloud;
}

bool savePointCloud(const std::string& filename, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) {
    if (pcl::io::savePLYFile<pcl::PointXYZRGB>(filename, *cloud) == -1) {
        PCL_ERROR("Couldn't write file %s \n", filename.c_str());
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <input_ply_file> <output_ply_file> <segment_length> <z_adjustment>" << std::endl;
        return -1;
    }

    std::string input_filename = argv[1];
    std::string output_filename = argv[2];
    float segment_length = std::stof(argv[3]);
    float z_adjustment = std::stof(argv[4]);

    auto cloud = loadPointCloud(input_filename);
    if (!cloud) {
        return -1;
    }

    // Apply z_adjustment
    Eigen::Matrix4f translation = Eigen::Matrix4f::Identity();
    translation(2, 3) = z_adjustment;
    pcl::transformPointCloud(*cloud, *cloud, translation);

    transformPointCloudIncrementally(cloud, segment_length);

    if (!savePointCloud(output_filename, cloud)) {
        return -1;
    }

    std::cout << "Transformation complete. Final output saved to " << output_filename << std::endl;
    return 0;
}
